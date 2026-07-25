import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import tempfile

# ==========================
# Load Model
# ==========================

model_name = "facebook/nllb-200-distilled-600M"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# ==========================
# Supported Languages
# ==========================

languages = {
    "Arabic": "arb_Arab",
    "English": "eng_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Spanish": "spa_Latn",
    "Italian": "ita_Latn",
    "Turkish": "tur_Latn",
    "Russian": "rus_Cyrl",
    "Chinese (Simplified)": "zho_Hans",
    "Japanese": "jpn_Jpan",
    "Korean": "kor_Hang",
    "Hindi": "hin_Deva",
    "Portuguese": "por_Latn",
    "Dutch": "nld_Latn",
    "Greek": "ell_Grek",
    "Polish": "pol_Latn",
    "Swedish": "swe_Latn",
}

# ==========================
# Translation Function
# ==========================

def translate(text, source_lang, target_lang):

    if not text.strip():
        return "Please enter text."

    tokenizer.src_lang = languages[source_lang]

    inputs = tokenizer(text, return_tensors="pt")

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(
            languages[target_lang]
        ),
        max_new_tokens=256,
    )

    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


def swap_languages(source_lang, target_lang):
    return target_lang, source_lang


def clear_fields():
    return "", "", "Characters: 0 | Words: 0", "Arabic", "English", None


def text_statistics(text):

    if not text.strip():
        return "Characters: 0 | Words: 0"

    characters = len(text)
    words = len(text.split())

    return f"Characters: {characters} | Words: {words}"


def download_translation(text):

    if not text.strip():
        return None

    file = tempfile.NamedTemporaryFile(
        delete=False, suffix=".txt", mode="w", encoding="utf-8"
    )

    file.write(text)
    file.close()

    return file.name


# ==========================
# Gradio Interface
# ==========================
with gr.Blocks(
    title="AI Multilingual Translator"
) as demo:
    gr.HTML("""
<div style="text-align:center; padding:20px;">
    <h1>🌍 AI Multilingual Translator</h1>
    <h3 style="color:gray;">
        Translate text between 200+ languages using Facebook NLLB-200
    </h3>
</div>
""")
    gr.Markdown("""
# 🤖 AI Multilingual Translator

### Fast, accurate, and AI-powered translation across 200+ languages
""")
    gr.Markdown("""
## Features

- Translate between 200+ languages
- Powered by Facebook NLLB
- Download translation
- Text statistics
- Fast AI Translation
""")
    with gr.Row():

        with gr.Column():

            input_text = gr.Textbox(
                lines=10,
                max_lines=15,
                label="📝 Enter Text",
                placeholder="Type or paste your text here...",
            )
            stats_box = gr.Textbox(label="📊 Text Statistics", interactive=False)
            input_text.change(fn=text_statistics, inputs=input_text, outputs=stats_box)

            source_dropdown = gr.Dropdown(
                choices=list(languages.keys()),
                value="Arabic",
                label="🌍 Source Language",
            )

            target_dropdown = gr.Dropdown(
                choices=list(languages.keys()),
                value="English",
                label="🎯 Target Language",
            )

    with gr.Row():

        translate_btn = gr.Button("🌍 Translate", variant="primary")
        swap_btn = gr.Button("🔄 Swap", size="lg")
        clear_btn = gr.Button("🧹 Clear", size="lg")
        download_btn = gr.Button("📥 Download TXT", size="lg")

        with gr.Column():

            output_text = gr.Textbox(
                lines=10,
                max_lines=15,
                label="🌍 Translation",
                interactive=False
            )
            download_file = gr.File(label="📄 Download Translation", interactive=False)

    translate_btn.click(
    fn=translate,
    inputs=[input_text, source_dropdown, target_dropdown],
    outputs=output_text,
    show_progress=True
)
    swap_btn.click(
        fn=swap_languages,
        inputs=[source_dropdown, target_dropdown],
        outputs=[source_dropdown, target_dropdown],
    )
    clear_btn.click(
        fn=clear_fields,
        inputs=[],
        outputs=[
            input_text,
            output_text,
            stats_box,
            source_dropdown,
            target_dropdown,
            download_file,
        ],
    )
    gr.Examples(
        examples=[
            ["أنا أحب البرمجة", "Arabic", "English"],
            ["كيف حالك؟", "Arabic", "French"],
            ["Hello World", "English", "Spanish"],
            ["Machine Learning", "English", "German"],
            ["Bonjour", "French", "Arabic"],
        ],
        inputs=[input_text, source_dropdown, target_dropdown],
    )
    gr.HTML("""
<div style="
display:flex;
justify-content:space-around;
margin-top:15px;">

<div style="
background:#EEF2FF;
padding:15px;
border-radius:12px;
width:30%;
text-align:center;">

<h3>🌐 Languages</h3>

<p>200+</p>

</div>

<div style="
background:#ECFDF5;
padding:15px;
border-radius:12px;
width:30%;
text-align:center;">

<h3>🤖 Model</h3>

<p>NLLB-200</p>

</div>

<div style="
background:#FEF3C7;
padding:15px;
border-radius:12px;
width:30%;
text-align:center;">

<h3>⚡ AI Powered</h3>

<p>Hugging Face</p>

</div>

</div>
""")
    gr.HTML("""
<hr>
<hr>
<div align="center">

AI Multilingual Translator

Powered by Facebook NLLB-200

Built with ❤️ using Gradio & Hugging Face

Developed by Habiba Gamal

</div>

<div style="text-align:center;color:gray;">

Developed using ❤️ Gradio & Hugging Face

<br>

Facebook NLLB-200-distilled-600M

</div>
""")
    download_btn.click(
        fn=download_translation,
        inputs=output_text,
        outputs=download_file,
    )

demo.launch(favicon_path="logo.png")
