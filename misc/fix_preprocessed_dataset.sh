root="/data/claire/preproc_ct/"
for file in "$root"/*png ; do
    name="$(basename "$file")"
    echo $name
    subj="$(echo "$name" | cut -d_ -f1)"
    sess="$(echo "$name" | cut -d_ -f2)"
    target_dir="$root"/"$subj"/"$sess"/mod-rx
    mkdir -p "$target_dir"
    mv "$file" "$target_dir"
done