from transformers import pipeline
import gradio as gr

pipe = pipeline(model="byoussef/whisper-large-v2-Ko")

def transcribe(audio, state=""):
    text = pipe(audio)["text"]
    state += text + " "
    return state, state
    
def main():
    iface = gr.Interface(
        fn=transcribe,
        inputs=[
            gr.Audio(source="microphone", type="filepath", streaming=True),
            "state"
        ],
        outputs=[
            "textbox",
            "state"
        ],
        live=True,
        title="Whisper Large-v2 Korean",
        description="Realtime demo for Korean speech recognition using a fine-tuned Whisper large-v2 model."
    )

    iface.launch(share=True)

if __name__ == "__main__":
    main()