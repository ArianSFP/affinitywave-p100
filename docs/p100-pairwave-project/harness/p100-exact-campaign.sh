#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
HARNESS="$SCRIPT_DIR/run-qualification.sh"
SUMMARY_HELPER="$SCRIPT_DIR/p100-exact-summary.py"
CAMPAIGNS_DIR="$SCRIPT_DIR/p100-exact-campaigns"

EXPECTED_LOGITS_SHA=47e84b679f12bae440e4f743d8305699a4f01cc9e22e18418824009f346020a5
EXPECTED_DUMP_FILES=640

MODE=list
CAMPAIGN_ID=
SEED=
PAIRS_2048=5
PAIRS_8128=3
COORDINATION_CONFIRMED=0
RUN_ACTIVE=0
EVENTS_FILE=

usage() {
    cat <<'EOF'
Usage:
  p100-exact-campaign.sh [list] [options]
  p100-exact-campaign.sh run --coordination-confirmed [options]
  p100-exact-campaign.sh verify --campaign ID
  p100-exact-campaign.sh summary --campaign ID

Modes:
  list       Print the randomized campaign and commands without running them.
             This is the default.
  run        Run exactness gates, then randomized OFF/ON performance pairs.
  verify     Recheck an existing campaign's c512 dumps and saved logits.
  summary    Summarize an existing campaign's paired timings without writing.

Options:
  --campaign ID             Safe campaign identifier.
  --seed SEED               Reproducible randomization seed.
  --pairs-2048 N            Number of pp2048 pairs (default: 5).
  --pairs-8128 N            Number of pp8128 pairs (default: 3).
  --coordination-confirmed  Confirm the required coordination entry was made.
  -h, --help                Show this help.

Run mode never overwrites a campaign directory or a runner artifact. Start a
new campaign ID after an interrupted run so the partial evidence is preserved.
Every GPU leg is delegated to run-qualification.sh, which owns the four-GPU
flock, process exclusion, production environment, and xsession watchdog.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 2
}

append_event() {
    local event=$1
    local detail=${2:-}

    if [[ -n "$EVENTS_FILE" ]]; then
        printf '%s\t%s\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$event" "$detail" >> "$EVENTS_FILE"
    fi
}

cleanup() {
    local status

    status=$1

    if [[ "$RUN_ACTIVE" -eq 1 ]]; then
        append_event campaign_aborted "status=$status"
    fi
}
trap 'cleanup $?' EXIT

validate_component() {
    local value=$1
    local label=$2

    if [[ ! "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
        die "$label must match [A-Za-z0-9][A-Za-z0-9._-]*"
    fi
}

validate_positive_integer() {
    local value=$1
    local label=$2

    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
        die "$label must be a positive integer"
    fi
}

validate_harness() {
    [[ -x "$HARNESS" ]] || die "qualification harness is not executable: $HARNESS"
    [[ -x "$SUMMARY_HELPER" ]] || die "summary helper is not executable: $SUMMARY_HELPER"

    if ! grep -Fq 'exec 9>/tmp/affinitywave-4gpu.lock' "$HARNESS" ||
       ! grep -Fq 'flock -n 9' "$HARNESS"; then
        die "qualification harness no longer contains the required four-GPU flock"
    fi
    if ! grep -Fq 'watch_xsession' "$HARNESS"; then
        die "qualification harness no longer contains the xsession watchdog"
    fi
}

arm_exact() {
    case "$1" in
        off) printf '0\n' ;;
        on)  printf '1\n' ;;
        *)   die "unknown arm: $1" ;;
    esac
}

pair_order() {
    local key=$1
    local checksum

    checksum=$(printf '%s' "${SEED}:${key}" | cksum)
    checksum=${checksum%% *}
    if (( checksum % 2 == 0 )); then
        printf 'off on\n'
    else
        printf 'on off\n'
    fi
}

emit_pair() {
    local stage=$1
    local tokens=$2
    local pair=$3
    local runner_mode=$4
    local order
    local first
    local second
    local position
    local arm
    local exact
    local tag

    order=$(pair_order "${stage}:${tokens}:${pair}")
    read -r first second <<< "$order"
    position=0
    for arm in "$first" "$second"; do
        position=$((position + 1))
        exact=$(arm_exact "$arm")
        tag="p100exact-${CAMPAIGN_ID}-${stage}-c${tokens}-p${pair}-${arm}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
            "$stage" "$tokens" "$pair" "$position" "$arm" "$exact" "$runner_mode" "$tag"
    done
}

emit_manifest() {
    local pair

    printf 'stage\ttokens\tpair\tposition\tarm\tp100_exact\trunner_mode\ttag\n'
    emit_pair dumps 512 1 diagonal-bench-dump
    emit_pair logits 512 1 diagonal-ppl-save
    for ((pair = 1; pair <= PAIRS_2048; pair++)); do
        emit_pair performance 2048 "$pair" diagonal-bench
    done
    for ((pair = 1; pair <= PAIRS_8128; pair++)); do
        emit_pair performance 8128 "$pair" diagonal-bench
    done
}

print_runner_command() {
    local runner_mode=$1
    local tag=$2
    local tokens=$3
    local exact=$4
    local command=(
        "$HARNESS"
        "$runner_mode"
        "$tag"
        "$tokens"
        2
        f32
        "$tokens"
        "$exact"
    )
    local argument

    printf '  '
    for argument in "${command[@]}"; do
        printf '%q ' "$argument"
    done
    printf '\n'
}

list_campaign() {
    local stage
    local tokens
    local pair
    local position
    local arm
    local exact
    local runner_mode
    local tag

    printf 'mode=list (no commands will run)\n'
    printf 'campaign=%s seed=%s pp2048_pairs=%s pp8128_pairs=%s\n' \
        "$CAMPAIGN_ID" "$SEED" "$PAIRS_2048" "$PAIRS_8128"
    printf 'expected_dump_files=%s expected_logits_sha=%s\n\n' \
        "$EXPECTED_DUMP_FILES" "$EXPECTED_LOGITS_SHA"
    emit_manifest
    printf '\nCommands, in execution order:\n'
    while IFS=$'\t' read -r stage tokens pair position arm exact runner_mode tag; do
        [[ "$stage" == stage ]] && continue
        print_runner_command "$runner_mode" "$tag" "$tokens" "$exact"
    done < <(emit_manifest)
    printf '\nRun mode also compares both dump trees, checks both logits files and their SHA-256,\n'
    printf 'then computes paired medians and a bootstrap 95%% lower bound.\n'
}

artifact_collision() {
    local tag=$1
    local candidate

    for candidate in \
        "$SCRIPT_DIR/$tag.out" \
        "$SCRIPT_DIR/$tag.err" \
        "$SCRIPT_DIR/$tag-dumps" \
        "$SCRIPT_DIR/$tag-logits-c512.bin" \
        "$SCRIPT_DIR/$tag.nsys-rep" \
        "$SCRIPT_DIR/$tag.sqlite" \
        "$SCRIPT_DIR/$tag.analysis.json" \
        "$SCRIPT_DIR/$tag.analysis.md"; do
        if [[ -e "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    return 1
}

check_artifact_collisions() {
    local stage
    local tokens
    local pair
    local position
    local arm
    local exact
    local runner_mode
    local tag
    local collision

    while IFS=$'\t' read -r stage tokens pair position arm exact runner_mode tag; do
        [[ "$stage" == stage ]] && continue
        if collision=$(artifact_collision "$tag"); then
            die "runner artifact already exists: $collision"
        fi
    done < <(emit_manifest)
}

manifest_tag() {
    local manifest=$1
    local stage=$2
    local arm=$3

    awk -F '\t' -v wanted_stage="$stage" -v wanted_arm="$arm" \
        'NR > 1 && $1 == wanted_stage && $5 == wanted_arm { print $8; exit }' \
        "$manifest"
}

verify_artifacts() {
    local campaign_dir=$1
    local manifest="$campaign_dir/manifest.tsv"
    local off_dump_tag
    local on_dump_tag
    local off_logits_tag
    local on_logits_tag
    local off_dump_dir
    local on_dump_dir
    local off_logits
    local on_logits
    local off_count
    local on_count
    local index
    local relative
    local mismatch=
    local off_sha=
    local on_sha=
    local status=0
    local -a off_files=()
    local -a on_files=()

    [[ -r "$manifest" ]] || die "missing campaign manifest: $manifest"
    off_dump_tag=$(manifest_tag "$manifest" dumps off)
    on_dump_tag=$(manifest_tag "$manifest" dumps on)
    off_logits_tag=$(manifest_tag "$manifest" logits off)
    on_logits_tag=$(manifest_tag "$manifest" logits on)
    [[ -n "$off_dump_tag" && -n "$on_dump_tag" ]] || die "manifest lacks dump arms"
    [[ -n "$off_logits_tag" && -n "$on_logits_tag" ]] || die "manifest lacks logits arms"

    off_dump_dir="$SCRIPT_DIR/$off_dump_tag-dumps"
    on_dump_dir="$SCRIPT_DIR/$on_dump_tag-dumps"
    off_logits="$SCRIPT_DIR/$off_logits_tag-logits-c512.bin"
    on_logits="$SCRIPT_DIR/$on_logits_tag-logits-c512.bin"

    printf 'check\tarm\tresult\tdetail\n'

    if [[ -d "$off_dump_dir" ]]; then
        mapfile -t off_files < <(
            cd -- "$off_dump_dir"
            find . -type f \( \
                -name '*-reduced-output-f32.bin' -o \
                -name '*-service-ids-i32.bin' -o \
                -name '*-service-input-f32.bin' -o \
                -name '*-service-weights-f32.bin' \
            \) -printf '%P\n' | LC_ALL=C sort
        )
    fi
    if [[ -d "$on_dump_dir" ]]; then
        mapfile -t on_files < <(
            cd -- "$on_dump_dir"
            find . -type f \( \
                -name '*-reduced-output-f32.bin' -o \
                -name '*-service-ids-i32.bin' -o \
                -name '*-service-input-f32.bin' -o \
                -name '*-service-weights-f32.bin' \
            \) -printf '%P\n' | LC_ALL=C sort
        )
    fi
    off_count=${#off_files[@]}
    on_count=${#on_files[@]}

    if [[ "$off_count" -eq "$EXPECTED_DUMP_FILES" ]]; then
        printf 'dump_count\toff\tPASS\tfiles=%s\n' "$off_count"
    else
        printf 'dump_count\toff\tFAIL\tfiles=%s expected=%s\n' "$off_count" "$EXPECTED_DUMP_FILES"
        status=1
    fi
    if [[ "$on_count" -eq "$EXPECTED_DUMP_FILES" ]]; then
        printf 'dump_count\ton\tPASS\tfiles=%s\n' "$on_count"
    else
        printf 'dump_count\ton\tFAIL\tfiles=%s expected=%s\n' "$on_count" "$EXPECTED_DUMP_FILES"
        status=1
    fi

    if [[ "$off_count" -ne "$on_count" ]]; then
        printf 'dump_layout\tboth\tFAIL\toff_files=%s on_files=%s\n' "$off_count" "$on_count"
        status=1
    else
        for ((index = 0; index < off_count; index++)); do
            if [[ "${off_files[index]}" != "${on_files[index]}" ]]; then
                mismatch="off=${off_files[index]} on=${on_files[index]}"
                break
            fi
        done
        if [[ -n "$mismatch" ]]; then
            printf 'dump_layout\tboth\tFAIL\t%s\n' "$mismatch"
            status=1
        else
            printf 'dump_layout\tboth\tPASS\tidentical relative paths\n'
            for relative in "${off_files[@]}"; do
                if ! cmp -s -- "$off_dump_dir/$relative" "$on_dump_dir/$relative"; then
                    mismatch=$relative
                    break
                fi
            done
            if [[ -n "$mismatch" ]]; then
                printf 'dump_bytes\tboth\tFAIL\tfirst_mismatch=%s\n' "$mismatch"
                status=1
            else
                printf 'dump_bytes\tboth\tPASS\tfiles=%s\n' "$off_count"
            fi
        fi
    fi

    if [[ -s "$off_logits" ]]; then
        off_sha=$(sha256sum -- "$off_logits")
        off_sha=${off_sha%% *}
    fi
    if [[ -s "$on_logits" ]]; then
        on_sha=$(sha256sum -- "$on_logits")
        on_sha=${on_sha%% *}
    fi
    if [[ "$off_sha" == "$EXPECTED_LOGITS_SHA" ]]; then
        printf 'logits_sha\toff\tPASS\t%s\n' "$off_sha"
    else
        printf 'logits_sha\toff\tFAIL\tactual=%s expected=%s\n' \
            "${off_sha:-missing}" "$EXPECTED_LOGITS_SHA"
        status=1
    fi
    if [[ "$on_sha" == "$EXPECTED_LOGITS_SHA" ]]; then
        printf 'logits_sha\ton\tPASS\t%s\n' "$on_sha"
    else
        printf 'logits_sha\ton\tFAIL\tactual=%s expected=%s\n' \
            "${on_sha:-missing}" "$EXPECTED_LOGITS_SHA"
        status=1
    fi
    if [[ -s "$off_logits" && -s "$on_logits" ]] &&
       cmp -s -- "$off_logits" "$on_logits"; then
        printf 'logits_bytes\tboth\tPASS\tbyte-identical\n'
    else
        printf 'logits_bytes\tboth\tFAIL\tnot byte-identical\n'
        status=1
    fi

    return "$status"
}

run_leg() {
    local stage=$1
    local tokens=$2
    local pair=$3
    local position=$4
    local arm=$5
    local exact=$6
    local runner_mode=$7
    local tag=$8

    append_event leg_start \
        "stage=$stage tokens=$tokens pair=$pair position=$position arm=$arm tag=$tag"
    if ! "$HARNESS" "$runner_mode" "$tag" "$tokens" 2 f32 "$tokens" "$exact"; then
        append_event leg_failed \
            "stage=$stage tokens=$tokens pair=$pair position=$position arm=$arm tag=$tag"
        return 1
    fi
    append_event leg_complete \
        "stage=$stage tokens=$tokens pair=$pair position=$position arm=$arm tag=$tag"
}

run_manifest_stage() {
    local manifest=$1
    local wanted_stage=$2
    local stage
    local tokens
    local pair
    local position
    local arm
    local exact
    local runner_mode
    local tag

    while IFS=$'\t' read -r stage tokens pair position arm exact runner_mode tag; do
        [[ "$stage" == "$wanted_stage" ]] || continue
        run_leg "$stage" "$tokens" "$pair" "$position" "$arm" "$exact" "$runner_mode" "$tag"
    done < "$manifest"
}

write_metadata() {
    local destination=$1

    {
        printf 'key\tvalue\n'
        printf 'campaign\t%s\n' "$CAMPAIGN_ID"
        printf 'seed\t%s\n' "$SEED"
        printf 'created_utc\t%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        printf 'pairs_2048\t%s\n' "$PAIRS_2048"
        printf 'pairs_8128\t%s\n' "$PAIRS_8128"
        printf 'expected_dump_files\t%s\n' "$EXPECTED_DUMP_FILES"
        printf 'expected_logits_sha\t%s\n' "$EXPECTED_LOGITS_SHA"
        printf 'qualification_harness\t%s\n' "$HARNESS"
    } > "$destination"
}

run_campaign() {
    local campaign_dir="$CAMPAIGNS_DIR/$CAMPAIGN_ID"
    local manifest="$campaign_dir/manifest.tsv"
    local exactness="$campaign_dir/exactness.tsv"
    local summary="$campaign_dir/summary.md"

    [[ "$COORDINATION_CONFIRMED" -eq 1 ]] ||
        die "run mode requires --coordination-confirmed after appending the active coordination file"
    validate_harness
    check_artifact_collisions

    mkdir -p -- "$CAMPAIGNS_DIR"
    if ! mkdir -- "$campaign_dir"; then
        die "campaign directory already exists; choose a new ID: $campaign_dir"
    fi
    EVENTS_FILE="$campaign_dir/events.tsv"
    printf 'timestamp_utc\tevent\tdetail\n' > "$EVENTS_FILE"
    RUN_ACTIVE=1

    emit_manifest > "$manifest"
    write_metadata "$campaign_dir/metadata.tsv"
    append_event campaign_start "seed=$SEED"

    run_manifest_stage "$manifest" dumps
    run_manifest_stage "$manifest" logits
    if ! verify_artifacts "$campaign_dir" > "$exactness"; then
        append_event exactness_failed "report=$exactness"
        printf 'Exactness gate failed; performance legs were not run. See %s\n' "$exactness" >&2
        return 1
    fi
    append_event exactness_passed "report=$exactness"

    run_manifest_stage "$manifest" performance
    if ! "$SUMMARY_HELPER" --gate "$campaign_dir" "$SCRIPT_DIR" > "$summary"; then
        append_event performance_gate_failed "summary=$summary"
        cat -- "$summary"
        return 1
    fi
    append_event campaign_complete "summary=$summary"
    RUN_ACTIVE=0
    cat -- "$summary"
}

resolve_existing_campaign() {
    local campaign_dir

    [[ -n "$CAMPAIGN_ID" ]] || die "$MODE mode requires --campaign ID"
    validate_component "$CAMPAIGN_ID" campaign
    campaign_dir="$CAMPAIGNS_DIR/$CAMPAIGN_ID"
    [[ -d "$campaign_dir" ]] || die "campaign does not exist: $campaign_dir"
    printf '%s\n' "$campaign_dir"
}

if [[ $# -gt 0 && "$1" != --* ]]; then
    MODE=$1
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --campaign)
            [[ $# -ge 2 ]] || die "--campaign requires an argument"
            CAMPAIGN_ID=$2
            shift 2
            ;;
        --seed)
            [[ $# -ge 2 ]] || die "--seed requires an argument"
            SEED=$2
            shift 2
            ;;
        --pairs-2048)
            [[ $# -ge 2 ]] || die "--pairs-2048 requires an argument"
            PAIRS_2048=$2
            shift 2
            ;;
        --pairs-8128)
            [[ $# -ge 2 ]] || die "--pairs-8128 requires an argument"
            PAIRS_8128=$2
            shift 2
            ;;
        --coordination-confirmed)
            COORDINATION_CONFIRMED=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            die "unknown argument: $1"
            ;;
    esac
done

case "$MODE" in
    list|dry-run)
        MODE=list
        CAMPAIGN_ID=${CAMPAIGN_ID:-preview}
        SEED=${SEED:-preview}
        validate_component "$CAMPAIGN_ID" campaign
        validate_component "$SEED" seed
        validate_positive_integer "$PAIRS_2048" pairs-2048
        validate_positive_integer "$PAIRS_8128" pairs-8128
        list_campaign
        ;;
    run)
        CAMPAIGN_ID=${CAMPAIGN_ID:-"$(date -u '+%Y%m%d-%H%M%S')-$$"}
        if [[ -z "$SEED" ]]; then
            SEED=$(od -An -N4 -tu4 /dev/urandom)
            SEED=${SEED//[[:space:]]/}
        fi
        validate_component "$CAMPAIGN_ID" campaign
        validate_component "$SEED" seed
        validate_positive_integer "$PAIRS_2048" pairs-2048
        validate_positive_integer "$PAIRS_8128" pairs-8128
        run_campaign
        ;;
    verify)
        campaign_dir=$(resolve_existing_campaign)
        verify_artifacts "$campaign_dir"
        ;;
    summary)
        campaign_dir=$(resolve_existing_campaign)
        validate_harness
        "$SUMMARY_HELPER" "$campaign_dir" "$SCRIPT_DIR"
        ;;
    *)
        die "unknown mode: $MODE"
        ;;
esac
