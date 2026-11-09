
#!/usr/bin/env bash
set -Eeuo pipefail

# GROMACS 2026.3 peptide-residue vs LPSA analysis

GMX="gmx_mpi"
TPR="MDrun1.tpr"
XTC="MDrun1_whole_pbc.xtc"
NDX="TAT_residues.ndx"
CUTOFF="0.35"
ROOT="TAT_LPS_contacts"

RES_NAMES=(
    GLY1 ARG2 LYS3 LYS4 ARG5 ARG6
    GLN7 ARG8 ARG9 ARG10 PRO11 GLN12
)
RES_IDS=(30 31 32 33 34 35 36 37 38 39 40 41)

LPS_NAMES=(ECLIP AKDO AHEP AGLC AGAL)
LPS_IDS=(13 14 15 16 17)

# Validate inputs
for file in "$TPR" "$XTC" "$NDX"; do
    if [[ ! -s "$file" ]]; then
        echo "ERROR: Missing or empty input file: $file" >&2
        exit 1
    fi
done

mkdir -p "$ROOT/contacts" "$ROOT/mindist" "$ROOT/logs"

TOTAL=$((${#RES_NAMES[@]} * ${#LPS_NAMES[@]}))
pair=0
skipped=0
calculated=0
failed=0

for i in "${!RES_NAMES[@]}"; do
    residue="${RES_NAMES[$i]}"
    rid="${RES_IDS[$i]}"

    for j in "${!LPS_NAMES[@]}"; do
        component="${LPS_NAMES[$j]}"
        lid="${LPS_IDS[$j]}"
        pair=$((pair + 1))

        contact_file="$ROOT/contacts/${residue}_${component}_contacts.xvg"
        distance_file="$ROOT/mindist/${residue}_${component}_mindist.xvg"
        log_file="$ROOT/logs/${residue}_${component}.log"

        # Skip pairs with both valid existing outputs
        if [[ -s "$contact_file" && -s "$distance_file" ]]; then
            skipped=$((skipped + 1))
            continue
        fi

        # Remove incomplete outputs before retrying
        rm -f "$contact_file" "$distance_file"

        if printf '%s\n%s\n' "$rid" "$lid" |
            "$GMX" mindist \
                -s "$TPR" \
                -f "$XTC" \
                -n "$NDX" \
                -on "$contact_file" \
                -od "$distance_file" \
                -d "$CUTOFF" \
                -tu ns \
                -xvg none >"$log_file" 2>&1; then

            if [[ -s "$contact_file" && -s "$distance_file" ]]; then
                calculated=$((calculated + 1))
                printf '[%d/%d] %s vs %s - SUCCESS\n' \
                    "$pair" "$TOTAL" "$residue" "$component"
            else
                failed=$((failed + 1))
                printf 'ERROR: [%d/%d] %s vs %s - empty output; see %s\n' \
                    "$pair" "$TOTAL" "$residue" "$component" "$log_file" >&2
            fi
        else
            failed=$((failed + 1))
            printf 'ERROR: [%d/%d] %s vs %s - failed; see %s\n' \
                "$pair" "$TOTAL" "$residue" "$component" "$log_file" >&2
        fi
    done
done

echo
echo "========== SUMMARY =========="
echo "Total pairs            : $TOTAL"
echo "Newly calculated       : $calculated"
echo "Skipped existing pairs : $skipped"
echo "Failed pairs           : $failed"
echo "Contacts folder        : $ROOT/contacts/"
echo "Distance folder        : $ROOT/mindist/"
echo "Logs folder            : $ROOT/logs/"
echo "=============================="

if (( failed > 0 )); then
    exit 1
fi
