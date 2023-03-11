from transformers import pipeline
import gradio as gr

pipe = pipeline(model="byoussef/whisper-large-v2-Ko")

def transcribe(audio):
    text = pipe(audio)["text"]
    return text
def main():
    iface = gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(source="microphone", type="filepath"),
        outputs="text",
        title="Whisper Large-v2 Korean",
        description="Realtime demo for Korean speech recognition using a fine-tuned Whisper large-v2 model.",
    )

    iface.launch(share=True)

if __name__ == "__main__":
    main()