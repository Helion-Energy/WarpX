"""Audit every generated arithmetic command; this is not runtime qualification.

The separate closure receipt binds fresh objects, archive members and the link.
"""

import argparse
import hashlib
import json
import os
import shlex
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def expanded(arguments, directory, depth=0):
    if depth > 8:
        raise ValueError("nested response-file depth")
    result = []
    index = 0
    while index < len(arguments):
        arg = arguments[index]
        response = None
        if arg.startswith("@"):
            response = arg[1:]
        elif arg in ("--options-file", "-optf"):
            index += 1
            response = arguments[index]
        elif arg.startswith("--options-file="):
            response = arg.split("=", 1)[1]
        if response is not None:
            for name in response.split(","):
                path = Path(name)
                if not path.is_absolute():
                    path = directory / path
                result.extend(
                    expanded(shlex.split(path.read_text()), directory, depth + 1)
                )
        elif arg in ("-Xcompiler", "--compiler-options"):
            index += 1
            result.extend(
                token
                for part in arguments[index].split(",")
                for token in shlex.split(part)
            )
        elif arg.startswith(("-Xcompiler=", "--compiler-options=")):
            result.extend(
                token
                for part in arg.split("=", 1)[1].split(",")
                for token in shlex.split(part)
            )
        else:
            result.append(arg)
        index += 1
    return result


def audit(context, records):
    cuda = Path(context["cuda_compiler"]).resolve()
    host = Path(context["host_compiler"]).resolve()
    if digest(cuda) != context["cuda_sha256"] or digest(host) != context["host_sha256"]:
        raise ValueError("compiler executable changed after configure")
    for name in (
        "NVCC_PREPEND_FLAGS",
        "NVCC_APPEND_FLAGS",
        "NVCC_CCBIN",
        "CUDAFLAGS",
        "CXXFLAGS",
        "CFLAGS",
    ):
        if os.environ.get(name):
            raise ValueError("unrecorded environment flags: " + name)
    if not records:
        raise ValueError("empty compilation closure")
    required_host = {
        "-fno-fast-math",
        "-fno-associative-math",
        "-fno-finite-math-only",
        "-ffp-contract=off",
    }
    required_cuda = {
        "--ftz=false",
        "--prec-div=true",
        "--prec-sqrt=true",
        "--fmad=false",
    }
    forbidden = {
        "-Ofast",
        "-ffast-math",
        "-funsafe-math-optimizations",
        "-fassociative-math",
        "-ffinite-math-only",
        "-freciprocal-math",
        "-fno-signed-zeros",
        "-fno-trapping-math",
        "-fcx-limited-range",
        "-ffp-contract=fast",
        "-ffp-contract=on",
        "-fexcess-precision=fast",
    }
    rows = []
    core = {"amrex": 0, "ablastr": 0, "warpx": 0}
    for record in records:
        directory = Path(record["directory"]).resolve()
        raw = record.get("arguments") or shlex.split(record["command"])
        args = expanded(raw, directory)
        compiler = Path(args[0]).resolve()
        is_cuda = compiler == cuda
        if is_cuda:
            host_choices = []
            for i, arg in enumerate(args):
                if arg in ("-ccbin", "--compiler-bindir"):
                    host_choices.append(args[i + 1])
                elif arg.startswith(("-ccbin=", "--compiler-bindir=")):
                    host_choices.append(arg.split("=", 1)[1])
            if not host_choices or any(Path(x).resolve() != host for x in host_choices):
                raise ValueError("NVCC host compiler differs from the pinned host")
        if compiler not in (cuda, host, Path("/usr/bin/cc").resolve()):
            raise ValueError("unsupported compiler/launcher: " + str(compiler))
        for index, arg in enumerate(args):
            for name, expected in (
                ("ftz", "false"),
                ("prec-div", "true"),
                ("prec-sqrt", "true"),
                ("fmad", "false"),
            ):
                if arg in ("--" + name, "-" + name):
                    if index + 1 == len(args) or args[index + 1] != expected:
                        raise ValueError("conflicting device arithmetic option: " + arg)
                elif arg.startswith(("--" + name + "=", "-" + name + "=")):
                    if arg.split("=", 1)[1] != expected:
                        raise ValueError("conflicting device arithmetic option: " + arg)
            if (
                arg in forbidden
                or "use_fast_math" in arg
                or arg.startswith("-flto")
                or arg
                in (
                    "--ftz=true",
                    "-ftz=true",
                    "--prec-div=false",
                    "-prec-div=false",
                    "--prec-sqrt=false",
                    "-prec-sqrt=false",
                    "--fmad=true",
                    "-fmad=true",
                )
                or arg.startswith(("-D__FAST_MATH__", "-D__FINITE_MATH_ONLY__"))
            ):
                raise ValueError("unsupported arithmetic flag: " + arg)
        if not required_host.issubset(args):
            raise ValueError("missing precise host flags: " + record["file"])
        if is_cuda and not required_cuda.issubset(args):
            raise ValueError("missing precise device flags: " + record["file"])
        source = Path(record["file"]).resolve()
        source_root = Path(context["source"]).resolve()
        amrex_root = Path(context["amrex"]).resolve()
        if source.is_relative_to(amrex_root):
            group = "amrex"
        elif source.is_relative_to(source_root / "Source/ablastr"):
            group = "ablastr"
        elif source.is_relative_to(source_root / "Source"):
            group = "warpx"
        else:
            group = "support"
        if group in core:
            core[group] += 1
        if "-o" not in args:
            raise ValueError("missing object output")
        output = Path(args[args.index("-o") + 1])
        if not output.is_absolute():
            output = directory / output
        if not output.resolve().is_relative_to(Path(context["binary"]).resolve()):
            raise ValueError("object outside fresh build: " + str(output))
        generated = source.is_relative_to(Path(context["binary"]).resolve())
        if not generated and not source.is_file():
            raise ValueError("missing immutable source: " + str(source))
        rows.append(
            {
                "source": str(source),
                "source_sha256": None if generated else digest(source),
                "generated_source_bound_by_postbuild_closure": generated,
                "object": str(output.resolve()),
                "group": group,
                "arguments": args,
            }
        )
    if not all(core.values()):
        raise ValueError("incomplete AMReX/ABLASTR/WarpX closure: " + str(core))
    return rows, core


def main():
    parser = argparse.ArgumentParser()
    for name in ("context", "commands", "header", "receipt"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    context = json.loads(args.context.read_text())
    rows, core = audit(context, json.loads(args.commands.read_text()))
    receipt = {
        "status": "PRECISE_COMMAND_CONTRACT_PASS_NOT_RUNTIME_QUALIFICATION",
        "context": context,
        "context_sha256": digest(args.context),
        "commands_sha256": digest(args.commands),
        "core_counts": core,
        "rows": rows,
    }
    encoded = json.dumps(receipt, indent=2) + "\n"
    receipt_hash = hashlib.sha256(encoded.encode()).hexdigest()
    header = (
        "// Generated only after the complete command audit.\n#pragma once\n"
        "namespace warpx::darwin {\n"
        "inline constexpr bool NativePairedPreciseCudaBuild = true;\n"
        f'inline constexpr char NativePairedPreciseCudaCommandReceipt[] = "{receipt_hash}";\n'
        "}\n"
    )
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.header.parent.mkdir(parents=True, exist_ok=True)
    for path, text in ((args.receipt, encoded), (args.header, header)):
        if not path.exists() or path.read_text() != text:
            path.write_text(text)
    print("Paired precise CUDA command audit:", core, receipt_hash)


if __name__ == "__main__":
    main()
