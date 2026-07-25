import tempfile
import gradio as gr
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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

  if text.strip() == "":
    return "Please enter text."

  tokenizer.src_lang = languages[source_lang]

  encoded = tokenizer(text, return_tensors="pt")

  generated_tokens = model.generate(
      **encoded,
      forced_bos_token_id=tokenizer.convert_tokens_to_ids(
          languages[target_lang]
      ),
      max_length=256,
  )

  translated = tokenizer.batch_decode(
      generated_tokens, skip_special_tokens=True
  )[0]

  return translated


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

with gr.Blocks(theme=gr.themes.Soft()) as demo:

  gr.Markdown("# 🌍 AI Multilingual Translator")
  gr.Markdown("Translate text between 200+ languages using Facebook NLLB-200")

  with gr.Row():
    # العمود الأيمن: النص المدخل وخيارات اللغات
    with gr.Column(scale=1):
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

    # العمود الأيسر: النص المترجم والملف
    with gr.Column(scale=1):
      output_text = gr.Textbox(
          lines=10, max_lines=15, label="🌍 Translation", interactive=False
      )
      download_file = gr.File(label="📄 Download Translation", interactive=False)

  # صف الأزرار للتحكم
  with gr.Row():
    translate_btn = gr.Button("🌍 Translate", variant="primary", size="lg")
    swap_btn = gr.Button("🔄 Swap", size="lg")
    clear_btn = gr.Button("🧹 Clear", size="lg")
    download_btn = gr.Button("📥 Download TXT", size="lg")

  # ربط الأحداث بالأزرار
  translate_btn.click(
      fn=translate,
      inputs=[input_text, source_dropdown, target_dropdown],
      outputs=output_text,
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
  download_btn.click(
      fn=download_translation,
      inputs=output_text,
      outputs=download_file,
  )

  # أمثلة للاستخدام
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

  # العناصر التجميلية والمعلوماتية (HTML)
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
    <div style="text-align:center;color:gray;">
    Developed using ❤️ Gradio & Hugging Face
    <br>
    Facebook NLLB-200-distilled-600M
    </div>
    """)

demo.launch()
