#!/usr/bin/env bash
# SPDX-Licence-Identifier: EUPL-1.2
#
# CLI artifact helpers for the lthn-rocm binary. Each command test builds and
# runs the binary in its own process, then asserts by exit code plus jq/grep.

run_capture_stdout() {
	local expected_status="$1"
	local output_file="$2"
	shift 2
	set +e
	"$@" >"$output_file"
	local status=$?
	set -e
	if [[ "$status" -ne "$expected_status" ]]; then
		printf 'expected exit %s, got %s\n' "$expected_status" "$status" >&2
		[[ -s "$output_file" ]] && { printf 'stdout:\n' >&2; cat "$output_file" >&2; }
		return 1
	fi
}

run_capture_all() {
	local expected_status="$1"
	local output_file="$2"
	shift 2
	set +e
	"$@" >"$output_file" 2>&1
	local status=$?
	set -e
	if [[ "$status" -ne "$expected_status" ]]; then
		printf 'expected exit %s, got %s\n' "$expected_status" "$status" >&2
		[[ -s "$output_file" ]] && { printf 'output:\n' >&2; cat "$output_file" >&2; }
		return 1
	fi
}

assert_jq() { jq -e "$1" "$2" >/dev/null; }
assert_contains() { grep -Fq "$1" "$2"; }

# Echoes the go-rocm module root (go/) from tests/cli/lthn-rocm/<command>/.
go_root() { ( cd "${1:-.}/../../../.." && pwd ); }

repo_root() { ( cd "$(go_root "${1:-.}")/.." && pwd ); }

build_lthn_rocm() {
	local root="$1"
	local bin="$root/tests/cli/lthn-rocm/bin/lthn-rocm"
	mkdir -p "$(dirname "$bin")"
	( cd "$root" && CGO_ENABLED="${CGO_ENABLED:-1}" go build -o "$bin" ./cmd/lthn-rocm ) >&2 || return 1
	printf '%s\n' "$bin"
}

hf_model_path() {
	local repo="$1"
	local snaps="$HOME/.cache/huggingface/hub/models--${repo//\//--}/snapshots"
	[[ -d "$snaps" ]] || return 2
	local snap
	snap="$(find "$snaps" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)"
	[[ -n "$snap" ]] || return 2
	printf '%s\n' "$snap"
}

model_path() {
	local repo_or_path="$1"
	if [[ -d "$repo_or_path" ]]; then
		printf '%s\n' "$repo_or_path"
		return 0
	fi
	hf_model_path "$repo_or_path"
}

gemma4_qat_target_path() {
	model_path "${GO_ROCM_CLI_GEMMA4_TARGET:-/data/ai/models/mlx-community/gemma-4-e2b-it-6bit}" ||
		model_path "mlx-community/gemma-4-e2b-it-6bit" ||
		model_path "mlx-community/gemma-4-e2b-it-4bit"
}

gemma4_qat_draft_path() {
	model_path "${GO_ROCM_CLI_GEMMA4_DRAFT:-/data/ai/models/mlx-community/gemma-4-E2B-it-qat-assistant-6bit}" ||
		model_path "mlx-community/gemma-4-E2B-it-qat-assistant-6bit" ||
		model_path "mlx-community/gemma-4-E2B-it-assistant-bf16"
}

kernel_hsaco() {
	if [[ -n "${GO_ROCM_KERNEL_HSACO:-}" && -f "${GO_ROCM_KERNEL_HSACO:-}" ]]; then
		printf '%s\n' "$GO_ROCM_KERNEL_HSACO"
		return 0
	fi
	local root
	root="$(repo_root "${1:-.}")"
	for candidate in \
		"$root/build/kernels/rocm_kernels_gfx1100.hsaco" \
		"$root/build/kernels/rocm_kernels.hsaco"; do
		if [[ -f "$candidate" ]]; then
			printf '%s\n' "$candidate"
			return 0
		fi
	done
	return 2
}
