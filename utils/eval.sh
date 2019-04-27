#!/bin/sh
# %%shell
#!/bin/bash

ref_file="./data/iwslt/dev.en"
translated_file=$1

echo "Reference File: " $ref_file
echo "Translated File: " $translated_file

# echo "Reversing BPE..."
# cat $translated_file | sed -E 's/(\@\@ )|(\@\@ ?$)//g' > $translated_file

REFERENCE_FILE=$ref_file
TRANSLATED_FILE=$translated_file

# The model output is expected to be in a tokenized form. Note, that if you
# tokenized your inputs to the model, then simply joined each output token with
# whitespace you should get tokenized outputs from your model.
# i.e. each output token is separate by whitespace
# e.g. "My model 's output is interesting ."
perl "./utils/tokenizer/detokenizer.perl" -l en < "$TRANSLATED_FILE" > "$TRANSLATED_FILE.detok"

PARAMS=("-tok" "intl" "-l" "de-en" "$REFERENCE_FILE")
sacrebleu "${PARAMS[@]}" < "$TRANSLATED_FILE.detok"