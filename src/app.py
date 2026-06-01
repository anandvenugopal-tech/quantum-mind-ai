import gradio as gr
import os

sys.path.append('/content/drive/MyDrive/DataScience/molecular-ai-agent')

from src.utils.mol_utils import draw_molecule, get_basic_descriptors

def process_input(user_input, agent):
  if not user_input.strip():
    return "Please enter a SMILES string or chemistry question."

  result = agent.run(user_input)

  if result.get('error'):
    return result['error'], None, ""

  if result['route'] == 'gnn':
    smiles = result.get('smiles', '')
    gap = result.get('homo_lumo_gap', 'N/A')
    desc = result.get('descriptors', {})

    # Draw molecule
    img = draw_molecule(smiles)

    # Format prediction output
    output = f"""Molecule Analysis
─────────────────────────────
SMILES         : {smiles}
HOMO-LUMO Gap  : {gap} eV

Molecular Properties
─────────────────────────────
Molecular Weight : {desc.get('molecular_weight', 'N/A')} g/mol
Num Atoms        : {desc.get('num_atoms', 'N/A')}
Num Rings        : {desc.get('num_rings', 'N/A')}
LogP             : {desc.get('logP', 'N/A')}
H-Bond Donors    : {desc.get('num_h_donors', 'N/A')}
H-Bond Acceptors : {desc.get('num_h_acceptors', 'N/A')}
"""
    if gap > 8:
          interp = f"Gap of {round(gap,2)} eV — Very stable, chemically inert. Excellent insulator."
    elif gap > 6:
          interp = f"Gap of {round(gap,2)} eV — Stable molecule, low reactivity. Good insulator."
    elif gap > 4:
          interp = f"Gap of {round(gap,2)} eV — Moderate stability. Typical drug-like molecule."
    elif gap > 2:
          interp = f"Gap of {round(gap,2)} eV — Moderately reactive. Potential semiconductor material."
    else:
          interp = f"Gap of {round(gap,2)} eV — Highly reactive. Potential organic conductor or dye."
    return output, img, interp

  else:
    return result.get('answer', 'No response.'), None, ""

def create_app(agent):
  with gr.Blocks(title = 'QuantumMind - Molecular AI Agent') as app:

    gr.Markdown("""
    # QuantumMind AI
    Enter a SMILES string to predict molecular properties, or ask a chemistry question.
    """)

    with gr.Row():
      with gr.Column(scale = 2):
        user_input = gr.Textbox(
            label = 'Input',
            placeholder = 'Enter SMILES (e.g. CC(=O)O) or chemistry question',
            lines = 2
        )
        submit_btn = gr.Button("Analyse", variant = "primary")

        # gr.Examples(
        #     examples=[
        #         ['CC(=0)0'],
        #         ['C1CCCCC1'],
        #         ['c1ccccc1'],
        #         ['What is the HOMO-LUMO gap?'],
        #         ['What does a small HOMO-LUMO gap indicate?'],
        #         ['What is the QM9 dataset?'],
        #     ],
        #     inputs=user_input
        # )

      with gr.Column(scale=1):
        mol_image = gr.Image(label = 'Molecule Structure')

    with gr.Row():
      output_text = gr.Textbox(label = 'Prediction / Answer', lines = 12)

    with gr.Row():
      interpretation = gr.Textbox(label = 'Interpretation', lines = 2)

    submit_btn.click(
        fn = lambda x: process_input(x, agent),
        inputs = [user_input],
        outputs = [output_text, mol_image, interpretation]
    )

  return app
