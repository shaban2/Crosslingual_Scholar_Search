import gradio as gr

from config import DEFAULT_ENCODER, FIGURES_TOP_K, TOP_K, index_path

INDEX_PATH = index_path(DEFAULT_ENCODER)
from pipeline.retrieval import search
from pipeline.synthesis import assemble_context, generate_review


def generate_literature_review(query: str, k: int = TOP_K) -> tuple[str, list[str], str]:
    figure_results = search(query, k=FIGURES_TOP_K, modality="figure")
    mixed_results = search(query, k=k)

    if not mixed_results and not figure_results:
        return "No relevant sources found. Please ensure the index has been built.", [], ""

    text_results = [r for r in mixed_results if r["metadata"].get("modality") == "text"]
    transcript_results = [r for r in mixed_results if r["metadata"].get("modality") == "transcript"]

    figure_image_paths = {r["metadata"].get("image_path") for r in figure_results if r["metadata"].get("image_path")}
    text_results = [r for r in text_results if r["metadata"].get("image_path") not in figure_image_paths]

    context = assemble_context(text_results, figure_results, transcript_results)
    review = generate_review(query, context)

    figure_paths = [r["metadata"]["image_path"] for r in figure_results]
    metrics = (
        f"**Retrieval Breakdown:** "
        f"📄 {len(text_results)} text passages  |  "
        f"🎯 {len(figure_results)} figures (dedicated retrieval)  |  "
        f"🎤 {len(transcript_results)} transcript segments"
    )
    return review, figure_paths, metrics


def index_status() -> str:
    if not INDEX_PATH.exists():
        return "Index not found. Run `python build_index.py` first."
    return "Index ready."


with gr.Blocks(title="Multimodal Research Synthesizer") as demo:
    gr.Markdown(
        """
        # Multimodal Research Synthesizer
        Enter a research query below to generate a structured literature review 
        synthesized from text (papers), figures, and lecture transcripts.
        """
    )

    with gr.Row():
        query_input = gr.Textbox(
            label="Research Query",
            placeholder="e.g. Compare attention mechanisms in Mamba vs Transformer",
            lines=3,
            scale=4,
        )
        k_slider = gr.Slider(
            minimum=3, maximum=30, value=TOP_K, step=1,
            label="Number of sources to retrieve (K)",
        )

    with gr.Row():
        submit_btn = gr.Button("Generate Review", variant="primary", scale=1)

    loading = gr.HTML(
        value="""
        <div style="text-align:center;padding:20px">
            <div style="border:3px solid #e5e7eb;border-top:3px solid #2563eb;border-radius:50%;width:28px;height:28px;animation:spin 0.8s linear infinite;margin:0 auto 10px"></div>
            <span style="color:#6b7280;font-size:14px">Generating literature review...</span>
        </div>
        <style>@keyframes spin{to{transform:rotate(360deg)}}</style>
        """,
        visible=False,
    )

    status_bar = gr.Markdown(index_status())

    metrics_bar = gr.Markdown(visible=False)

    output = gr.Markdown(
        label="Literature Review",
        value="*Your generated literature review will appear here.*",
    )

    figure_gallery = gr.Gallery(
        label="Retrieved Figures",
        show_label=True,
        columns=3,
        height="auto",
        object_fit="contain",
    )

    def on_generate(query, k):
        if not query or not query.strip():
            return "Please enter a query.", gr.update(visible=False), [], ""
        if not INDEX_PATH.exists():
            return "Index not found. Run `python build_index.py` first.", gr.update(visible=False), [], ""
        review, figure_paths, metrics = generate_literature_review(query.strip(), int(k))
        return review, gr.update(visible=False), figure_paths, metrics

    submit_btn.click(
        fn=lambda q, k: (gr.update(visible=True), gr.update(visible=True), [], ""),
        inputs=[query_input, k_slider],
        outputs=[loading, output, figure_gallery, metrics_bar],
    ).then(
        fn=on_generate,
        inputs=[query_input, k_slider],
        outputs=[output, loading, figure_gallery, metrics_bar],
    )

    gr.Markdown(
        """
        ---
        ### How it works
        1. **Build index** — Run `python build_index.py` to process PDFs and YouTube transcripts into a FAISS vector index.
        2. **Query** — Type a research question above. The system retrieves relevant text passages, figures, and lecture clips.
        3. **Synthesize** — Llama 3.2 reads all retrieved sources together and generates a structured review with citations.
        """
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
