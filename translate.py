from transformers import MarianMTModel, MarianTokenizer
import sacrebleu
from striprtf.striprtf import rtf_to_text

# --------- FUNCTION: RTF -> CLEAN TEXT ----------
def load_rtf_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        rtf_content = f.read()
    
    text = rtf_to_text(rtf_content)

    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return lines

# --------- LOAD MODEL ----------
model_name = "Helsinki-NLP/opus-mt-bn-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# --------- LOAD INPUT ----------
bengali_sentences = load_rtf_lines("input.rtf")

# --------- TRANSLATE ----------
translated_sentences = []

for sentence in bengali_sentences:
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    translated = model.generate(**inputs)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)
    translated_sentences.append(output)

# --------- SAVE OUTPUT ----------
with open("output.txt", "w", encoding="utf-8") as f:
    for line in translated_sentences:
        f.write(line + "\n")

print("Translation completed. Saved to output.txt")

# --------- LOAD REFERENCE ----------
reference_sentences = load_rtf_lines("reference.rtf")

# --------- BLEU ----------
if len(translated_sentences) != len(reference_sentences):
    raise ValueError("Mismatch between number of translated and reference sentences!")

bleu = sacrebleu.corpus_bleu(translated_sentences, [reference_sentences])

print(f"\nBLEU Score: {bleu.score:.2f}")

# --------- FIRST SENTENCE ----------
if translated_sentences:
    print("\nFirst translated sentence:")
    print(translated_sentences[0])