"""HumanEval / MBPP / BigCodeBench code env: 给 problem -> 收 code -> 在 sandbox 跑 unit test.

每个 problem 是一个 task-level CP unit.

两个接口:
- step(code)         -> (success: bool, error_msg: str)
                        backward-compat, 给 cascade_runner / run_pipeline 等老调用方
- step_verdict(code) -> (verdict: str, stderr: str)
                        新接口, 给 multi-action router 数据收集用,
                        verdict ∈ {pass, fail, timeout, compile_error, infra_error}
"""
from __future__ import annotations

import base64
import json
import pickle
import re
import subprocess
import tempfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


# Verdict 枚举 (字符串, 不用 Enum 是为了 jsonl 序列化最干净).
# parse_error 不在这里产生 — 它是 agent 层在 LLM 输出无法解析时单独标的,
# 这里只关心 "代码已经送进 subprocess" 之后的结果分类.
VERDICT_PASS          = "pass"
VERDICT_FAIL          = "fail"
VERDICT_TIMEOUT       = "timeout"
VERDICT_COMPILE_ERROR = "compile_error"
VERDICT_INFRA_ERROR   = "infra_error"   # subprocess 自己挂了, 通常要 retry / drop

VERDICT_TYPES = frozenset({
    VERDICT_PASS, VERDICT_FAIL, VERDICT_TIMEOUT,
    VERDICT_COMPILE_ERROR, VERDICT_INFRA_ERROR,
})


# 行首匹配 "<错误类型>:" 才算 compile-time / import-time 错误.
# 这些错误意味着 "代码连测试都跑不到", 跟 AssertionError (跑到了但测失败) 不同.
_COMPILE_ERROR_REGEX = re.compile(
    r"^(SyntaxError|IndentationError|TabError|ImportError|ModuleNotFoundError):",
    re.MULTILINE,
)


def _classify_verdict(returncode: int, stderr: str) -> str:
    """Map (returncode, stderr) 到 verdict.

    timeout / infra_error 由 caller 在 except 分支单独设置; 这里只处理
    subprocess 已正常退出 (无论 0 或非 0) 的情况.
    """
    if returncode == 0:
        return VERDICT_PASS
    # 看 stderr 最后 ~10 行, 取最后一个匹配 compile-error 模式的行
    tail = "\n".join(stderr.strip().splitlines()[-10:])
    if _COMPILE_ERROR_REGEX.search(tail):
        return VERDICT_COMPILE_ERROR
    return VERDICT_FAIL

try:
    from datasets import load_dataset
except ImportError as e:
    load_dataset = None
    _IMPORT_ERR = e
else:
    _IMPORT_ERR = None


@dataclass
class CodeProblem:
    task_id: str
    prompt: str           # function signature + docstring
    test: str             # 单元 test (humaneval 风 / unittest 风, 看 test_format)
    entry_point: str      # 函数名
    test_format: str = "humaneval"   # "humaneval" 或 "unittest"


def _strip_code_block(text: str) -> str:
    """Agent 输出可能带 ```python ``` 包裹, 抽出来."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


def _build_full_program(code: str, test: str, entry_point: str, test_format: str) -> str:
    if test_format == "humaneval":
        return f"{code}\n\n{test}\n\ncheck({entry_point})\n"
    if test_format == "unittest":
        # BigCodeBench: test 文件已经定义 class TestCases(unittest.TestCase)
        return (
            f"{code}\n\n{test}\n\n"
            "import unittest, sys\n"
            "suite = unittest.TestLoader().loadTestsFromTestCase(TestCases)\n"
            "result = unittest.TextTestRunner(verbosity=0, stream=sys.stderr).run(suite)\n"
            "sys.exit(0 if result.wasSuccessful() else 1)\n"
        )
    if test_format == "unittest_main":
        # ClassEval-style: test file has multiple TestCase subclasses, no single 'TestCases'
        # name. Use unittest.main with loader discovery to run them all.
        return (
            f"{code}\n\n{test}\n\n"
            "import unittest, sys\n"
            "loader = unittest.TestLoader()\n"
            "suite = unittest.TestSuite()\n"
            "for name, obj in list(globals().items()):\n"
            "    if isinstance(obj, type) and issubclass(obj, unittest.TestCase) and obj is not unittest.TestCase:\n"
            "        suite.addTests(loader.loadTestsFromTestCase(obj))\n"
            "result = unittest.TextTestRunner(verbosity=0, stream=sys.stderr).run(suite)\n"
            "sys.exit(0 if result.wasSuccessful() else 1)\n"
        )
    if test_format == "livecodebench_functional":
        # LeetCode-style: test 已经包含完整 assertion loop (调 Solution().method()),
        # 直接拼到 code 后面跑即可.
        return f"{code}\n\n{test}\n"
    if test_format == "ds1000":
        # DS-1000: test 字段是 code_context 全文, 提供 test_execution(solution: str).
        # 调 test_execution 用 repr() 把 model code 包成 Python literal string.
        return f"{test}\n\ntest_execution({code!r})\n"
    if test_format == "scicode_e2e":
        # SciCode: test 字段已 preprocess (target = refN injected). 内含一个
        # marker `# === MODEL CODE INSERTED ABOVE ===` 替换为 model code.
        marker = "# === MODEL CODE INSERTED ABOVE ==="
        if marker in test:
            return test.replace(marker, code) + "\n"
        # fallback: append code after test (less ideal)
        return f"{test}\n\n{code}\n"
    if test_format == "apps_functional":
        # APPS functional-style: test is a JSON payload {"fn_name": str, "io": [[in, out], ...]}.
        # We synthesize a Python test harness that:
        #   1) calls fn_name with input, trying f(*input) first then f(input) as fallback
        #      (per APPS official evaluator's heterogeneous-input compatibility shim).
        #   2) compares result against expected output (deep equality after normalization).
        # This handles the ~3300 APPS-train functional-style problems with fn_name set.
        return (
            f"{code}\n\n"
            "import json as _apps_json\n"
            "import sys as _apps_sys\n"
            f"_apps_payload = _apps_json.loads({json.dumps(test)})\n"
            "_apps_fn_name = _apps_payload['fn_name']\n"
            "_apps_fn = globals().get(_apps_fn_name)\n"
            "if _apps_fn is None:\n"
            "    print(f'NameError: function {_apps_fn_name} not defined', file=_apps_sys.stderr)\n"
            "    _apps_sys.exit(1)\n"
            "def _apps_norm(x):\n"
            "    if isinstance(x, tuple):\n"
            "        return list(x)\n"
            "    return x\n"
            "def _apps_call_compat(fn, inp):\n"
            "    # Try f(*inp), then f(inp). Returns (ok, result_or_exc).\n"
            "    if isinstance(inp, list):\n"
            "        try:\n"
            "            return True, fn(*inp)\n"
            "        except (TypeError, ValueError):\n"
            "            pass\n"
            "    try:\n"
            "        return True, fn(inp)\n"
            "    except Exception as e:\n"
            "        return False, e\n"
            "for _apps_i, (_apps_in, _apps_exp) in enumerate(_apps_payload['io']):\n"
            "    _apps_ok, _apps_got = _apps_call_compat(_apps_fn, _apps_in)\n"
            "    if not _apps_ok:\n"
            "        raise AssertionError(f'Test {_apps_i}: fn raised {type(_apps_got).__name__}: {_apps_got}')\n"
            "    # APPS outputs are sometimes wrapped in a one-elem list — unwrap.\n"
            "    _apps_expected = _apps_exp[0] if (isinstance(_apps_exp, list) and len(_apps_exp) == 1) else _apps_exp\n"
            "    if _apps_norm(_apps_got) != _apps_norm(_apps_expected):\n"
            "        raise AssertionError(f'Test {_apps_i}: expected {_apps_expected!r}, got {_apps_got!r}')\n"
        )
    raise ValueError(f"unknown test_format: {test_format}")


def _run_debugbench_lc_test(code: str, test_payload_json: str,
                             timeout: float = 30.0) -> Tuple[str, str]:
    """DebugBench: submit `code` to LeetCode via leetcode_env package.
    Requires env vars LEETCODE_SESSION + LEETCODE_CSRF_TOKEN.

    Returns (verdict, stderr) where verdict is:
      - pass: Accepted
      - fail: Wrong Answer / Time Limit Exceeded / Memory Limit / Output Limit
      - compile_error: Compile Error / Runtime Error (pre-test)
      - infra_error: anything else (network, auth, rate-limit)
    """
    import os
    if not os.environ.get("LEETCODE_SESSION") or not os.environ.get("LEETCODE_CSRF_TOKEN"):
        return VERDICT_INFRA_ERROR, ("RUNNER_ERROR: LEETCODE_SESSION/LEETCODE_CSRF_TOKEN "
                                     "env vars not set — DebugBench requires LC auth.")
    try:
        from leetcode_env.types import LeetCodeSubmission, ProgrammingLanguage
        from leetcode_env.environment import LeetCodeEnv
    except ImportError as e:
        return VERDICT_INFRA_ERROR, f"RUNNER_ERROR: leetcode_env not installed: {e}"

    try:
        payload = json.loads(test_payload_json)
    except Exception as e:
        return VERDICT_INFRA_ERROR, f"RUNNER_ERROR: bad test payload: {e}"
    slug = payload.get("slug", "")
    if not slug:
        return VERDICT_INFRA_ERROR, "RUNNER_ERROR: no LC slug in test payload"

    clean_code = _strip_code_block(code)
    sub = LeetCodeSubmission(
        code=clean_code,
        lang=ProgrammingLanguage.PYTHON3,
        question_slug=slug,
        timeout=int(timeout),
    )
    env = LeetCodeEnv()
    try:
        status, reward, done, result = env.step(sub)
    except Exception as e:
        return VERDICT_INFRA_ERROR, f"RUNNER_ERROR: LC submit failed: {type(e).__name__}: {str(e)[:300]}"

    # Map LC status to our verdict
    status_msg = (result or {}).get("status_msg", "") if isinstance(result, dict) else str(status)
    status_low = status_msg.lower()
    if "accepted" in status_low:
        return VERDICT_PASS, ""
    if "compile error" in status_low or "compilation error" in status_low:
        err = (result or {}).get("compile_error", "") if isinstance(result, dict) else ""
        return VERDICT_COMPILE_ERROR, f"LC compile error: {err}"[:1500]
    if "runtime error" in status_low:
        err = (result or {}).get("runtime_error", "") if isinstance(result, dict) else ""
        # treat runtime error as compile_error-ish (didn't reach assertion logic)
        return VERDICT_COMPILE_ERROR, f"LC runtime error: {err}"[:1500]
    if "time limit" in status_low or "memory limit" in status_low or "output limit" in status_low:
        return VERDICT_TIMEOUT, f"LC {status_msg}"[:1500]
    if "wrong answer" in status_low:
        # include the failed input/output for reflect to use
        details = ""
        if isinstance(result, dict):
            ex_input = result.get("input", "") or result.get("last_testcase", "")
            ex_expected = result.get("expected_output", "")
            ex_actual = result.get("code_output", "") or result.get("std_output", "")
            details = f"Wrong Answer | input: {ex_input!r} | expected: {ex_expected!r} | got: {ex_actual!r}"
        return VERDICT_FAIL, details[:1500] or f"LC: {status_msg}"
    # unknown
    return VERDICT_INFRA_ERROR, f"LC unknown status: {status_msg}"


def _run_lcb_stdin_tests(code: str, test_cases_json: str,
                         timeout: float = 30.0) -> Tuple[str, str]:
    """LiveCodeBench / TACO stdin-style: 每个 test case 单独 subprocess 喂 stdin.

    性能优化: 通过 env var LCB_MAX_TEST_CASES 限制每题最多跑多少个 test
    (default 20). TACO 平均 94 cases/题, 限到 20 提速 ~4x, label noise 极小.
    设为 0 表示不限制.
    """
    import os as _os
    try:
        test_cases = json.loads(test_cases_json)
    except Exception as e:
        return VERDICT_INFRA_ERROR, f"RUNNER_ERROR: bad test_cases_json: {e}"

    # 限制 test cases 数, 加速 rollout (默认 20)
    max_tests = int(_os.environ.get("LCB_MAX_TEST_CASES", "20"))
    if max_tests > 0:
        test_cases = test_cases[:max_tests]

    # 一次性写 code 到临时文件 — 多 test 共用
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            tmp_path = f.name
    except Exception as e:
        return VERDICT_INFRA_ERROR, f"RUNNER_ERROR: {type(e).__name__}: {e}"

    try:
        for i, tc in enumerate(test_cases):
            try:
                result = subprocess.run(
                    ["python", tmp_path],
                    input=tc.get("input", ""),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return VERDICT_TIMEOUT, f"Test {i}: TimeoutError (>{timeout}s)"
            except Exception as e:
                return VERDICT_INFRA_ERROR, f"RUNNER_ERROR test {i}: {type(e).__name__}: {e}"

            if result.returncode != 0:
                stderr = (result.stderr or "").strip() or (result.stdout or "").strip()
                verdict = _classify_verdict(result.returncode, stderr)
                # 第一个 fail/compile_error 就直接返, 不继续跑后面 test
                snippet = (stderr or "")[:1500]
                return verdict, f"Test {i}: {snippet}"

            # Whitespace-normalized stdout 比较 (按 split() token-level 比, 容忍尾换行)
            actual_tokens = (result.stdout or "").strip().split()
            expected_tokens = tc.get("output", "").strip().split()
            if actual_tokens != expected_tokens:
                err = (f"Test {i}: expected {tc.get('output','')[:200]!r}, "
                       f"got {(result.stdout or '')[:200]!r}")
                return VERDICT_FAIL, err
        return VERDICT_PASS, ""
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _run_subprocess_typed(code: str, test: str, entry_point: str,
                          test_format: str = "humaneval",
                          timeout: float = 30.0) -> Tuple[str, str]:
    """exec(code + test) 在 subprocess 跑, 返回 (verdict, stderr).

    verdict ∈ {pass, fail, timeout, compile_error, infra_error}.

    test_format:
      - "humaneval":     test 是 `def check(candidate)`, 调 check(entry_point)
      - "unittest":      BigCodeBench, 跑 unittest.TestCase
      - "livecodebench_functional": LeetCode 风, test 含 Solution().method() 断言 loop
      - "livecodebench_stdin":      AtCoder/CF 风, 每个 test 单独 subprocess 喂 stdin
    """
    # stdin 路径走独立 runner (每 test 单 subprocess)
    if test_format == "livecodebench_stdin":
        return _run_lcb_stdin_tests(code, test, timeout)
    # DebugBench: 走 LC API (需要 LEETCODE_SESSION env var)
    if test_format == "debugbench_lc":
        return _run_debugbench_lc_test(code, test, timeout)

    full_program = _build_full_program(code, test, entry_point, test_format)

    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(full_program)
            tmp_path = f.name
        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        finally:
            Path(tmp_path).unlink(missing_ok=True)

        stderr = (result.stderr or "").strip()
        if not stderr and result.returncode != 0:
            # 某些失败把信息打到 stdout (e.g. unittest verbose=0 失败)
            stderr = (result.stdout or "").strip()
        verdict = _classify_verdict(result.returncode, stderr)
        return verdict, stderr[:2000]
    except subprocess.TimeoutExpired:
        return VERDICT_TIMEOUT, f"TimeoutError: execution exceeded {timeout}s"
    except Exception as e:
        return VERDICT_INFRA_ERROR, f"RUNNER_ERROR: {type(e).__name__}: {e}"


def _run_subprocess(code: str, test: str, entry_point: str,
                    test_format: str = "humaneval",
                    timeout: float = 30.0) -> Tuple[bool, str]:
    """Backward-compat shim: 老调用方拿 (success, error_msg)."""
    verdict, stderr = _run_subprocess_typed(code, test, entry_point, test_format, timeout)
    return (verdict == VERDICT_PASS), stderr


# ============================================================
# LiveCodeBench loader (drop-in code_generation_lite via hf_hub_download,
# 不走 datasets loading-script — 新 datasets library 已不支持)
# ============================================================

# 累积 release: release_vN 包含 test.jsonl 到 testN.jsonl
_LCB_RELEASE_FILES = {
    "release_v1": ["test.jsonl"],
    "release_v2": ["test.jsonl", "test2.jsonl"],
    "release_v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "release_v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "release_v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "release_v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}
_LCB_LATEST = "release_v6"
_LCB_REPO_ID = "livecodebench/code_generation_lite"


def _lcb_decode_private_tests(encoded: str) -> list:
    """LiveCodeBench private_test_cases 解码: base64 → zlib → pickle → JSON list."""
    if not encoded:
        return []
    decoded = pickle.loads(zlib.decompress(base64.b64decode(encoded)))
    return json.loads(decoded) if isinstance(decoded, str) else decoded


def _lcb_build_functional_test(test_cases: list, func_name: str) -> str:
    """LeetCode-style: 生成 inline assertion loop, 调 Solution().{func_name}(*args)."""
    # 嵌入 test_cases JSON 字面量, 避免 quote escaping
    tc_payload = json.dumps([
        {"input": tc["input"], "output": tc["output"]} for tc in test_cases
    ])
    return f'''
import json as __json__
__lcb_cases = __json__.loads({tc_payload!r})
for __i, __tc in enumerate(__lcb_cases):
    __input_str = __tc["input"]
    __expected = __json__.loads(__tc["output"])
    # multi-arg: input lines separated by \\n, 每行一个 JSON literal
    __arg_lines = __input_str.split("\\n") if "\\n" in __input_str else [__input_str]
    __args = [__json__.loads(__line) for __line in __arg_lines]
    try:
        __result = Solution().{func_name}(*__args)
    except Exception as __e:
        raise AssertionError(f"Test {{__i}} raised {{type(__e).__name__}}: {{__e}}")
    assert __result == __expected, (
        f"Test {{__i}}: expected {{__expected!r}}, got {{__result!r}}"
    )
'''


def _lcb_build_problem(row: dict) -> Optional["CodeProblem"]:
    """把 LiveCodeBench 一行原始 jsonl 转成 CodeProblem.
    返回 None 表示 row 不完整 (无 test cases / 缺 func_name 等) — 跳过.
    """
    pub = json.loads(row["public_test_cases"]) if row.get("public_test_cases") else []
    prv = _lcb_decode_private_tests(row.get("private_test_cases", ""))
    all_cases = pub + prv
    if not all_cases:
        return None
    testtype = all_cases[0].get("testtype", "stdin")
    qid = row["question_id"]

    if testtype == "functional":
        meta = json.loads(row.get("metadata") or "{}")
        func_name = meta.get("func_name", "")
        if not func_name:
            return None
        starter = (row.get("starter_code") or "").strip()
        prompt = (
            row["question_content"]
            + "\n\n## Starter code\n```python\n" + starter + "\n```\n\n"
            "Complete the Solution class. Output ONLY the full class definition "
            "(with `class Solution:`) plus any imports it needs."
        )
        test = _lcb_build_functional_test(all_cases, func_name)
        return CodeProblem(
            task_id=f"LiveCodeBench/{qid}",
            prompt=prompt,
            test=test,
            entry_point=func_name,
            test_format="livecodebench_functional",
        )

    # stdin (AtCoder / Codeforces)
    starter = (row.get("starter_code") or "").strip()
    prompt = row["question_content"]
    if starter:
        prompt += "\n\n## Starter code\n```python\n" + starter + "\n```"
    prompt += "\n\nWrite a complete Python program that reads from stdin and writes to stdout."
    test_payload = json.dumps([
        {"input": tc["input"], "output": tc["output"]} for tc in all_cases
    ])
    return CodeProblem(
        task_id=f"LiveCodeBench/{qid}",
        prompt=prompt,
        test=test_payload,
        entry_point="",
        test_format="livecodebench_stdin",
    )


# ============================================================
# TACO loader (BAAI/TACO, Dec 2023, 26k contest-style problems)
# arrow shards via hf_hub_download (datasets loading-script bypass).
# ============================================================

_TACO_REPO_ID = "BAAI/TACO"
_TACO_TRAIN_SHARDS = [f"train/data-{i:05d}-of-00009.arrow" for i in range(9)]
_TACO_INTERACTIVE_MARKERS = (
    "this is an interactive problem",
    "this is interactive",
)


def _taco_is_interactive(row: dict) -> bool:
    """Skip interactive problems — can't be tested with single-shot stdin/stdout."""
    raw_tags = (row.get("raw_tags") or "").lower()
    if "interactive" in raw_tags:
        return True
    q_head = (row.get("question") or "").lower()[:300]
    return any(m in q_head for m in _TACO_INTERACTIVE_MARKERS)


def _taco_build_problem(row: dict) -> Optional["CodeProblem"]:
    """Build CodeProblem from TACO row. Currently only stdin/stdout supported.

    Returns None for:
      - rows with no parseable input_output
      - rows with mismatched #inputs / #outputs
      - rows with fn_name set (functional format, not yet supported)
      - rows with `class Solution:` starter_code (LeetCode/GFG-style functional, format heterogeneous)
      - **rows from source=geeksforgeeks**: GFG inputs use human-readable variable
        assignments (e.g. 'n = 5\\nk = 3\\narr[] = {5,10,30,20,15}'), neither pure
        stdin nor JSON literal — would systematic-fail. Skip until proper parser.
      - rows with 0 test cases
    """
    # GFG 整个跳过 (input format 异构)
    if row.get("source") == "geeksforgeeks":
        return None

    io_str = row.get("input_output") or ""
    if not io_str:
        return None
    try:
        io_obj = json.loads(io_str)
    except Exception:
        return None
    ins = io_obj.get("inputs") or []
    outs = io_obj.get("outputs") or []
    if not ins or len(ins) != len(outs):
        return None
    fn_name = io_obj.get("fn_name")
    if fn_name:
        # TACO functional format: inputs are list-of-args; format heterogeneous,
        # 先跳过 functional, 只用 stdin/stdout 子集. 后面要支持再加.
        return None
    # Detect leftover Solution-class style (functional 但没标 fn_name) → 也跳
    starter = (row.get("starter_code") or "").strip()
    if "class Solution" in starter or "class Solution:" in starter:
        return None

    # 一些行的 inputs/outputs 是 list 嵌套 (有的 wrap, 有的不), 全统一成 string
    def _to_str(x):
        if isinstance(x, str):
            return x
        return "\n".join(str(e) for e in x) if isinstance(x, list) else str(x)

    test_cases = [
        {"input": _to_str(inp), "output": _to_str(out)}
        for inp, out in zip(ins, outs)
    ]
    test_payload = json.dumps(test_cases)

    # task_id: 用 URL 最后 2 段保唯一 (e.g. codeforces "1063/C" → "1063_C",
    # 否则不同 contest 同字母题会 collide). fallback 用 name.
    url = (row.get("url") or "").rstrip("/")
    if url:
        parts = url.split("/")
        slug = "_".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
    else:
        slug = row.get("name") or "?"
    source = row.get("source") or "unknown"
    qid = f"TACO/{source}/{slug}"

    prompt = row["question"]
    starter = (row.get("starter_code") or "").strip()
    if starter:
        prompt += "\n\n## Starter code\n```python\n" + starter + "\n```"
    prompt += "\n\nWrite a complete Python program that reads from stdin and writes to stdout."

    return CodeProblem(
        task_id=qid,
        prompt=prompt,
        test=test_payload,
        entry_point="",
        test_format="livecodebench_stdin",   # 复用 stdin 执行路径
    )


def _load_taco(difficulty_filter=("HARD", "VERY_HARD"),
               limit: Optional[int] = None,
               exclude_interactive: bool = True) -> List["CodeProblem"]:
    """加载 TACO train shards, filter by difficulty (default: HARD + VERY_HARD)."""
    try:
        from huggingface_hub import hf_hub_download
        from datasets import Dataset
    except ImportError as e:
        raise ImportError("TACO loader 需要 huggingface_hub + datasets") from e

    diff_set = set(difficulty_filter) if difficulty_filter else None
    problems: List[CodeProblem] = []
    for shard in _TACO_TRAIN_SHARDS:
        path = hf_hub_download(repo_id=_TACO_REPO_ID, filename=shard, repo_type="dataset")
        ds = Dataset.from_file(path)
        for row in ds:
            if diff_set and row.get("difficulty") not in diff_set:
                continue
            if exclude_interactive and _taco_is_interactive(row):
                continue
            prob = _taco_build_problem(row)
            if prob is None:
                continue
            problems.append(prob)
            if limit and len(problems) >= limit:
                return problems
    return problems


def _load_livecodebench(version_tag: str = _LCB_LATEST,
                        date_after: Optional[str] = None,
                        difficulty: Optional[str] = None,
                        limit: Optional[int] = None) -> List["CodeProblem"]:
    """加载 LiveCodeBench 题目.

    Args:
      version_tag:   "release_v1" .. "release_v6" (default v6 = latest)
      date_after:    ISO date "YYYY-MM-DD"; 只保留 contest_date >= 该日的 (default None = 全收)
      difficulty:    "easy" / "medium" / "hard"; 默认全收
      limit:         最多多少题; 默认全收

    Returns: list[CodeProblem]
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError("LiveCodeBench loader 需要 huggingface_hub: pip install huggingface_hub") from e

    files = _LCB_RELEASE_FILES.get(version_tag, _LCB_RELEASE_FILES[_LCB_LATEST])
    problems: List[CodeProblem] = []
    seen_ids: set = set()  # 去重 — cumulative release 同题可能出现多次
    for fname in files:
        local = hf_hub_download(repo_id=_LCB_REPO_ID, filename=fname, repo_type="dataset")
        with open(local) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if date_after and (row.get("contest_date", "")[:10] < date_after):
                    continue
                if difficulty and row.get("difficulty") != difficulty:
                    continue
                qid = row.get("question_id")
                if qid in seen_ids:
                    continue
                seen_ids.add(qid)
                prob = _lcb_build_problem(row)
                if prob is None:
                    continue
                problems.append(prob)
                if limit and len(problems) >= limit:
                    return problems
    return problems


# ============================================================
# DS-1000 loader (xlangai/DS-1000, 1000 Python data-science problems)
# Test format: each row has `code_context` with a `test_execution(solution: str)` fn.
# ============================================================

def _load_ds1000(limit: Optional[int] = None) -> List["CodeProblem"]:
    """Load DS-1000 (xlangai/DS-1000) Python data-science problems.
    All problems are functional-style; tests live in code_context.
    """
    from datasets import load_dataset as _ld
    ds = _ld("xlangai/DS-1000", split="test")
    problems: List[CodeProblem] = []
    for row in ds:
        meta = row.get("metadata") or {}
        prob_id = meta.get("problem_id", "?")
        lib = meta.get("library", "?")
        prompt = (
            "## Problem\n" + row["prompt"]
            + "\n\nWrite a Python solution. The final result must be stored in "
              "a variable named `result`. Output ONLY the solution code "
              "(no imports of the wrapper, no explanations)."
        )
        problems.append(CodeProblem(
            task_id=f"DS-1000/{lib}/{prob_id}",
            prompt=prompt,
            test=row["code_context"],   # full test driver lives here
            entry_point="",              # not used for ds1000 test_format
            test_format="ds1000",
        ))
        if limit and len(problems) >= limit:
            break
    return problems


# ============================================================
# SciCode loader (SciCode1/SciCode, 80 sci-computing problems)
# Test format: deps + N sub-step impls + general_tests (uses `target` placeholder).
# ============================================================

# SciCode test_data.h5 path. Download from Google Drive (~1GB):
#   https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR
# Save to data/scicode/SciCode_test/test_data.h5 (relative to repo root) or via env var.
_SCICODE_H5_DEFAULT = "data/scicode/SciCode_test/test_data.h5"


def _scicode_h5_path() -> str:
    """Resolve test_data.h5 path. Allows SCICODE_H5_PATH env override."""
    import os as _os
    return _os.environ.get("SCICODE_H5_PATH", _SCICODE_H5_DEFAULT)


def _scicode_build_test_script(sub_steps: list, h5_path: str) -> str:
    """Build the test prelude/suite script. Per-sub-step:
      targets = process_hdf5_to_tuple('P.S', N)
      target = targets[0]; <test_case 0>
      target = targets[1]; <test_case 1>
      ...
    Model code is inserted at the # === MODEL CODE === marker.
    """
    prelude = (
        "from scicode.parse.parse import process_hdf5_to_tuple\n"
        f"_H5_PATH = {h5_path!r}\n"
        "\n# === MODEL CODE INSERTED ABOVE ===\n\n"
    )
    parts = [prelude]
    for ss in sub_steps:
        step_id = ss.get("step_number")
        tcs = ss.get("test_cases") or []
        if not tcs or not step_id:
            continue
        parts.append(f"# --- Sub-step {step_id} ({len(tcs)} test cases) ---\n")
        parts.append(f"targets = process_hdf5_to_tuple({step_id!r}, {len(tcs)}, h5py_file=_H5_PATH)\n")
        for idx, tc in enumerate(tcs):
            parts.append(f"target = targets[{idx}]\n")
            parts.append(tc.rstrip() + "\n\n")
    parts.append("print('SCICODE_ALL_TESTS_PASSED')\n")
    return "".join(parts)


def _load_scicode(split: str = "test", limit: Optional[int] = None) -> List["CodeProblem"]:
    """Load SciCode (SciCode1/SciCode) scientific-computing problems.

    Tests at the sub-step level using `process_hdf5_to_tuple` from
    `scicode.parse.parse` (needs `pip install -e <SciCode repo>` + h5 file
    downloaded from Google Drive — see SCICODE_H5_PATH env var).
    """
    from datasets import load_dataset as _ld
    ds = _ld("SciCode1/SciCode", split=split)
    h5_path = _scicode_h5_path()
    problems: List[CodeProblem] = []
    for row in ds:
        pid = row["problem_id"]
        pname = row["problem_name"]
        deps = row.get("required_dependencies") or ""
        sub_steps = row.get("sub_steps") or []
        # Build problem prompt
        sub_step_text = "\n\n".join(
            f"### Sub-step {ss['step_number']}\n{ss['step_description_prompt']}\n"
            f"```python\n{ss.get('function_header','')}\n```"
            for ss in sub_steps
        )
        prompt = (
            f"## Problem: {pname}\n\n"
            + row.get("problem_description_main", "")
            + "\n\n## IO spec\n" + row.get("problem_io", "")
            + "\n\n## Required dependencies (already imported)\n```python\n" + deps + "\n```"
            + "\n\n## Sub-steps to implement\n" + sub_step_text
            + "\n\nImplement ALL sub-step functions. Output ONLY the complete Python "
              "code (functions only, with the required imports). The dependencies "
              "above are already imported in the test environment."
        )
        # Build composite test script — deps + h5 reader + per-sub-step tests
        test_script = deps + "\n\n" + _scicode_build_test_script(sub_steps, h5_path)
        problems.append(CodeProblem(
            task_id=f"SciCode/{pid}/{pname}",
            prompt=prompt,
            test=test_script,
            entry_point="",
            test_format="scicode_e2e",
        ))
        if limit and len(problems) >= limit:
            break
    return problems


# ============================================================
# HumanEvalFix Python loader (bigcode/humanevalpack, Python subset, 164 bug-fix tasks)
# Test format: humaneval (uses check(entry_point) pattern from canonical HE).
# ============================================================

def _load_humanevalfix(limit: Optional[int] = None) -> List["CodeProblem"]:
    """Load HumanEvalFix Python (bigcode/humanevalpack 'python' config, 'test' split).
    Task: given buggy_solution + prompt + tests, fix the bug.
    For our cascade: ignore buggy_solution, ask model to write from scratch.
    """
    from datasets import load_dataset as _ld
    ds = _ld("bigcode/humanevalpack", "python", split="test")
    problems: List[CodeProblem] = []
    for row in ds:
        prompt = (
            row["prompt"] + "\n"
            + row.get("docstring", "")
            + "\n\n# Buggy version provided for reference (DO NOT just submit this):\n"
            + row.get("buggy_solution", "")
            + "\n\nWrite the corrected function. Output ONLY imports + the complete "
              "function definition."
        )
        problems.append(CodeProblem(
            task_id=f"HEFix/{row['task_id']}",
            prompt=prompt,
            test=row["test"],
            entry_point=row["entry_point"],
            test_format="humaneval",
        ))
        if limit and len(problems) >= limit:
            break
    return problems


# ============================================================
# DebugBench Python loader (Rtian/DebugBench, ~1.4k Python LC problems)
# Test runner: REQUIRES LeetCode session (leetcode_env package).
# 这里只 load problems; verdict 路径需要单独配 LC env, 详见 verify_debugbench() stub.
# ============================================================

def _load_debugbench_python(limit: Optional[int] = None) -> List["CodeProblem"]:
    """Load DebugBench Python subset. Tests run via leetcode_env (user provides LC cookie).
    For local stub testing without LC: uses the bundled `solution` as ground truth
    and runs model code + GT on parsed `examples` text, compares stdout.
    """
    from datasets import load_dataset as _ld
    ds = _ld("Rtian/DebugBench", split="test")
    problems: List[CodeProblem] = []
    for row in ds:
        if row.get("language") != "python3":
            continue
        prompt = (
            row["question"]
            + "\n\n# Examples:\n" + str(row.get("examples", ""))
            + "\n\n# Constraints:\n" + str(row.get("constraints", ""))
            + "\n\n# Reference buggy code (DO NOT just submit):\n"
            + row.get("buggy_code", "")
            + "\n\nWrite a Solution class with the required method. Output ONLY the "
              "Python class definition + any imports."
        )
        # test field stores LC slug + ground-truth for downstream leetcode_env runner
        test_payload = json.dumps({
            "slug": row["slug"],
            "ground_truth": row["solution"],
            "category": row["category"],
            "subtype": row["subtype"],
        })
        problems.append(CodeProblem(
            task_id=f"DebugBench/{row['slug']}",
            prompt=prompt,
            test=test_payload,
            entry_point="",
            test_format="debugbench_lc",   # ⚠ NOT YET IMPLEMENTED in _run_subprocess_typed
        ))
        if limit and len(problems) >= limit:
            break
    return problems


# ============================================================
# APPS (Hendrycks et al. 2021) — 5k test problems, 3 difficulty tiers
# ============================================================

_APPS_REPO_ID = "codeparrot/apps"


def _apps_build_problem(row: dict, split: str = "test") -> Optional["CodeProblem"]:
    """APPS row → CodeProblem. Skip functional-style (fn_name set) and class-Solution starters.

    task_id format: "APPS/<split>/<difficulty>/<id>" — split prefix avoids id collision
    between train (ids 0..4999) and test (ids 0..4999).
    """
    io_raw = row.get("input_output") or ""
    if not io_raw:
        return None
    try:
        io_obj = json.loads(io_raw) if isinstance(io_raw, str) else io_raw
    except Exception:
        return None
    ins = io_obj.get("inputs") or []
    outs = io_obj.get("outputs") or []
    if not ins or len(ins) != len(outs):
        return None
    if io_obj.get("fn_name"):
        return None  # skip functional, format heterogeneous
    starter = (row.get("starter_code") or "").strip()
    if "class Solution" in starter:
        return None

    def _to_str(x):
        if isinstance(x, str):
            return x
        return "\n".join(str(e) for e in x) if isinstance(x, list) else str(x)

    test_cases = [{"input": _to_str(inp), "output": _to_str(out)}
                  for inp, out in zip(ins, outs)]
    test_payload = json.dumps(test_cases)

    prompt = row["question"]
    if starter:
        prompt += "\n\n## Starter code\n```python\n" + starter + "\n```"
    prompt += "\n\nWrite a complete Python program that reads from stdin and writes to stdout."

    qid = f"APPS/{split}/{row.get('difficulty', '?')}/{row['id']}"
    return CodeProblem(
        task_id=qid,
        prompt=prompt,
        test=test_payload,
        entry_point="",
        test_format="livecodebench_stdin",
    )


def _load_apps(difficulty_filter: Optional[Tuple[str, ...]] = None,
               limit: Optional[int] = None,
               splits: Tuple[str, ...] = ("test", "train")) -> List["CodeProblem"]:
    """Load APPS problems from one or both splits (default: both = ~10k problems).

    Args:
      difficulty_filter: subset of {'introductory','interview','competition'}.
      limit: cap total problems returned.
      splits: which split files to read. APPS = 5000 train + 5000 test.
    """
    from huggingface_hub import hf_hub_download
    diff_set = set(difficulty_filter) if difficulty_filter else None
    problems: List[CodeProblem] = []
    for split in splits:
        fname = f"{split}.jsonl"
        path = hf_hub_download(_APPS_REPO_ID, fname, repo_type="dataset")
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if diff_set and row.get("difficulty") not in diff_set:
                    continue
                p = _apps_build_problem(row, split=split)
                if p is None:
                    continue
                problems.append(p)
                if limit and len(problems) >= limit:
                    return problems
    return problems


def _apps_functional_build_problem(row: dict, split: str = "test") -> Optional["CodeProblem"]:
    """APPS functional-style row → CodeProblem. Only rows with fn_name set + non-Solution starter.
    Test runner = `apps_functional` (synthesizes a call-compat harness).
    """
    io_raw = row.get("input_output") or ""
    if not io_raw:
        return None
    try:
        io_obj = json.loads(io_raw) if isinstance(io_raw, str) else io_raw
    except Exception:
        return None
    fn_name = io_obj.get("fn_name")
    if not fn_name:
        return None   # not functional-style; handled by _apps_build_problem
    ins = io_obj.get("inputs") or []
    outs = io_obj.get("outputs") or []
    if not ins or len(ins) != len(outs):
        return None
    starter = (row.get("starter_code") or "").strip()
    if "class Solution" in starter:
        return None   # LC-style class wrappers — too brittle, skip

    # Cap test count to keep wall + payload reasonable
    io_pairs = list(zip(ins[:20], outs[:20]))
    test_payload = json.dumps({"fn_name": fn_name, "io": io_pairs})

    prompt = row["question"]
    if starter:
        prompt += "\n\n## Starter code\n```python\n" + starter + "\n```"
    prompt += (
        f"\n\nDefine a Python function named `{fn_name}` (and any helpers it needs) "
        "at module level. The function will be called directly with the test inputs. "
        "Output only valid Python — imports + function definition(s)."
    )

    qid = f"APPS/{split}/{row.get('difficulty', '?')}_func/{row['id']}"
    return CodeProblem(
        task_id=qid,
        prompt=prompt,
        test=test_payload,
        entry_point=fn_name,
        test_format="apps_functional",
    )


def _load_apps_functional(difficulty_filter: Optional[Tuple[str, ...]] = None,
                          limit: Optional[int] = None,
                          splits: Tuple[str, ...] = ("test", "train")) -> List["CodeProblem"]:
    """Load APPS functional-style problems (those skipped by _load_apps due to fn_name)."""
    from huggingface_hub import hf_hub_download
    diff_set = set(difficulty_filter) if difficulty_filter else None
    problems: List[CodeProblem] = []
    for split in splits:
        fname = f"{split}.jsonl"
        path = hf_hub_download(_APPS_REPO_ID, fname, repo_type="dataset")
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if diff_set and row.get("difficulty") not in diff_set:
                    continue
                p = _apps_functional_build_problem(row, split=split)
                if p is None:
                    continue
                problems.append(p)
                if limit and len(problems) >= limit:
                    return problems
    return problems


# ============================================================
# CodeContests (DeepMind 2022) — 165 test + 13k train competitive problems
# ============================================================

_CC_REPO_ID = "deepmind/code_contests"
_CC_TEST_PARQUET = "data/test-00000-of-00001-9c49eeff30aacaa8.parquet"


def _cc_build_problem(row: dict, max_tests: int = 20) -> Optional["CodeProblem"]:
    """CodeContests row → CodeProblem. Use public + private tests (skip massive generated_tests)."""
    pub = row.get("public_tests") or {}
    priv = row.get("private_tests") or {}
    all_in = list(pub.get("input") or []) + list(priv.get("input") or [])
    all_out = list(pub.get("output") or []) + list(priv.get("output") or [])
    if not all_in or len(all_in) != len(all_out):
        return None
    # Cap test count to avoid mega-cases blowing up rollout wall
    all_in = all_in[:max_tests]
    all_out = all_out[:max_tests]
    test_cases = [{"input": i, "output": o} for i, o in zip(all_in, all_out)]
    test_payload = json.dumps(test_cases)

    prompt = (row.get("description") or "").strip()
    if not prompt:
        return None
    prompt += "\n\nWrite a complete Python program that reads from stdin and writes to stdout."

    qid = f"CodeContests/{row.get('name', '?').replace(' ', '_').replace('/', '_')[:80]}"
    return CodeProblem(
        task_id=qid,
        prompt=prompt,
        test=test_payload,
        entry_point="",
        test_format="livecodebench_stdin",
    )


def _load_codecontests(limit: Optional[int] = None,
                       split: str = "test") -> List["CodeProblem"]:
    """Load CodeContests test split (165 problems). split='train' would need shard iter."""
    from huggingface_hub import hf_hub_download
    from datasets import Dataset
    if split != "test":
        raise NotImplementedError("only test split implemented; train has 39 shards")
    path = hf_hub_download(_CC_REPO_ID, _CC_TEST_PARQUET, repo_type="dataset")
    ds = Dataset.from_parquet(path)
    problems: List[CodeProblem] = []
    for row in ds:
        p = _cc_build_problem(row)
        if p is None:
            continue
        problems.append(p)
        if limit and len(problems) >= limit:
            break
    return problems


# ============================================================
# ClassEval (Fudan SE Lab 2023) — 100 class-level Python generation problems
# ============================================================

_CE_REPO_ID = "FudanSELab/ClassEval"
_CE_TEST_PARQUET = "data/test-00000-of-00001-5c45fa6e45572491.parquet"


def _load_classeval(limit: Optional[int] = None) -> List["CodeProblem"]:
    """ClassEval: write a Python class given a skeleton + description. Tests run via unittest."""
    from huggingface_hub import hf_hub_download
    from datasets import Dataset
    path = hf_hub_download(_CE_REPO_ID, _CE_TEST_PARQUET, repo_type="dataset")
    ds = Dataset.from_parquet(path)
    problems: List[CodeProblem] = []
    for row in ds:
        prompt = (
            row["skeleton"]
            + "\n\n## Class description\n"
            + str(row.get("class_description", "")).strip()
        )
        prompt += (
            "\n\nImplement the class and all its methods. "
            "Output only valid Python code (imports + class definition)."
        )
        # test field is unittest-style — wrap in a check(candidate) shim per humaneval format,
        # OR run via the "unittest" format which runs the test code directly. Use unittest.
        test_code = row["test"]
        problems.append(CodeProblem(
            task_id=f"ClassEval/{row['task_id']}",
            prompt=prompt,
            test=test_code,
            entry_point=row.get("class_name", ""),
            test_format="unittest_main",
        ))
        if limit and len(problems) >= limit:
            break
    return problems


class CodeEnv:
    """单 dataset 的 code env."""

    def __init__(self, dataset: str = "humaneval", split: str = "test",
                 limit: Optional[int] = None,
                 lcb_version: str = _LCB_LATEST,
                 lcb_date_after: Optional[str] = None,
                 lcb_difficulty: Optional[str] = None,
                 taco_difficulties: Optional[List[str]] = None,
                 apps_difficulties: Optional[List[str]] = None):
        """
        Args:
          dataset:   "humaneval" / "bigcodebench" / "mbpp" / "livecodebench" / "taco"
          split:     dataset 内部 split (livecodebench / taco 用自己的 filter)
          limit:     最多加载 N 题
          lcb_version:    LiveCodeBench release ("release_v1" .. "release_v6"; 默认 v6)
          lcb_date_after: LiveCodeBench 时间过滤 ISO "YYYY-MM-DD" (default None = 全收)
          lcb_difficulty: LiveCodeBench 难度过滤 "easy"/"medium"/"hard" (default None)
          taco_difficulties: TACO 难度过滤列表 (default ["HARD","VERY_HARD"])
        """
        if _IMPORT_ERR is not None and dataset not in ("livecodebench", "taco"):
            raise ImportError(
                "需要安装 datasets: pip install datasets"
            ) from _IMPORT_ERR

        if dataset == "humaneval":
            ds = load_dataset("openai/openai_humaneval", split=split)
            self.problems: List[CodeProblem] = [
                CodeProblem(
                    task_id=row["task_id"],
                    prompt=row["prompt"],
                    test=row["test"],
                    entry_point=row["entry_point"],
                    test_format="humaneval",
                )
                for row in ds
            ]
        elif dataset == "bigcodebench":
            # split: v0.1.0_hf / v0.1.1 / v0.1.2 / v0.1.3 / v0.1.4 都是 1140 题, 用最新版
            ds = load_dataset("bigcode/bigcodebench", split="v0.1.4")
            self.problems = [
                CodeProblem(
                    task_id=row["task_id"],
                    prompt=row["complete_prompt"],   # 含 import + signature + docstring
                    test=row["test"],                 # unittest.TestCase 类
                    entry_point=row["entry_point"],
                    test_format="unittest",
                )
                for row in ds
            ]
        elif dataset == "mbpp":
            ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)
            self.problems = [
                CodeProblem(
                    task_id=f"MBPP/{row['task_id']}",
                    # MBPP prompts are imperative ("Write a Python function to ..."); the
                    # required function name is implied by test_list (e.g. remove_Occ).
                    # Inline the test_list so model knows the expected signature.
                    prompt=(
                        row["prompt"]
                        + "\n\n# Your implementation must satisfy:\n"
                        + "\n".join(row["test_list"])
                    ),
                    # Test runs as plain top-level asserts after model code.
                    # Reuse livecodebench_functional format = `{code}\n\n{test}\n`.
                    test="\n".join(row["test_list"]),
                    entry_point="",
                    test_format="livecodebench_functional",
                )
                for row in ds
            ]
        elif dataset == "livecodebench":
            # LiveCodeBench code_generation_lite (release_v6 = 880+ 题, contest-style).
            # 通过 huggingface_hub 直接读 jsonl, 绕开 datasets loading-script 不再支持的问题.
            # 默认全收 release_v6; 可用 lcb_date_after="2024-10-01" 等过滤 post-cutoff 题.
            self.problems = _load_livecodebench(
                version_tag=lcb_version,
                date_after=lcb_date_after,
                difficulty=lcb_difficulty,
                limit=limit,   # 直接在 loader 里 short-circuit
            )
        elif dataset == "taco":
            # TACO (BAAI Dec 2023, 26k contest 题, train split). 默认只取 HARD+VERY_HARD.
            # 通过 hf_hub_download 直接读 arrow shards, 绕 loading-script.
            self.problems = _load_taco(
                difficulty_filter=tuple(taco_difficulties or ("HARD", "VERY_HARD")),
                limit=limit,
                exclude_interactive=True,
            )
        elif dataset == "ds1000":
            self.problems = _load_ds1000(limit=limit)
        elif dataset == "scicode":
            self.problems = _load_scicode(split=split or "test", limit=limit)
        elif dataset == "humanevalfix":
            self.problems = _load_humanevalfix(limit=limit)
        elif dataset == "debugbench":
            # ⚠ test runner via LC API not yet integrated; use loader for prompt loading only.
            self.problems = _load_debugbench_python(limit=limit)
        elif dataset == "apps":
            self.problems = _load_apps(
                difficulty_filter=tuple(apps_difficulties) if apps_difficulties else None,
                limit=limit,
            )
        elif dataset == "apps_functional":
            self.problems = _load_apps_functional(
                difficulty_filter=tuple(apps_difficulties) if apps_difficulties else None,
                limit=limit,
            )
        elif dataset == "codecontests":
            self.problems = _load_codecontests(limit=limit)
        elif dataset == "classeval":
            self.problems = _load_classeval(limit=limit)
        else:
            raise ValueError(
                f"unknown dataset: {dataset}. "
                "支持: humaneval / bigcodebench / mbpp / livecodebench / taco / "
                "ds1000 / scicode / humanevalfix / debugbench / apps / apps_functional / "
                "codecontests / classeval"
            )

        if limit and dataset not in ("livecodebench", "taco", "ds1000",
                                     "scicode", "humanevalfix", "debugbench",
                                     "apps", "apps_functional",
                                     "codecontests", "classeval"):
            # 上述 loaders 已在内部 limit 过
            self.problems = self.problems[:limit]

    def __len__(self):
        return len(self.problems)

    def get(self, idx: int) -> CodeProblem:
        return self.problems[idx]

    def step(self, problem: CodeProblem, code: str, timeout: float = 30.0):
        """跑 code + test, 返回 (success: bool, error_msg: str). Backward-compat."""
        clean_code = _strip_code_block(code)
        return _run_subprocess(
            clean_code, problem.test, problem.entry_point,
            test_format=problem.test_format, timeout=timeout,
        )

    def step_verdict(self, problem: CodeProblem, code: str,
                     timeout: float = 30.0) -> Tuple[str, str]:
        """跑 code + test, 返回 (verdict, stderr).

        verdict ∈ {pass, fail, timeout, compile_error, infra_error}.
        parse_error 不会从这里返回 — 那是 agent 层在 LLM 输出不可解析时标的.
        """
        clean_code = _strip_code_block(code)
        return _run_subprocess_typed(
            clean_code, problem.test, problem.entry_point,
            test_format=problem.test_format, timeout=timeout,
        )
