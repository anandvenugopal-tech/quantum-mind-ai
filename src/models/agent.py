from rdkit import Chem
import torch
from torch_geometric.data import Data
from src.utils.mol_utils import smiles_to_features, get_basic_descriptors

class MolecularAgent:

  MOLECULE_KEYWORDS = [
        'smiles', 'molecule', 'predict', 'property',
        'structure', 'compound', 'formula', 'atoms'
    ]

  CHEMISTRY_KEYWORDS = [
        'homo', 'lumo', 'gap', 'dipole', 'energy',
        'orbital', 'electron', 'aromatic', 'bond',
        'explain', 'what is', 'why', 'how does',
        'what does', 'define', 'meaning'
    ]

  def __init__(self, gnn_model = None, llm_pipeline = None, device = 'cpu'):
    self.gnn = gnn_model
    self.llm = llm_pipeline
    self.device = device

  def classify_input(self, user_input: str) -> str:
    text = user_input.lower().strip()
    has_smiles_chars = any(c in user_input for c in ['=', '#', '[', ']'])
    has_mol_keyword = any(kw in text for kw in self.MOLECULE_KEYWORDS)
    if has_smiles_chars or has_mol_keyword:
      return 'gnn'
    has_chem_keyword = any(kw in text for kw in self.CHEMISTRY_KEYWORDS)
    if has_chem_keyword:
      return 'llm'
    return 'llm'

  def extract_smiles(self, user_input: str):
    words = user_input.split()
    for word in words:
      word = word.strip('.,?!')
      if any(c in word for c in ['=', '#', '(', ')', '[', ']']):
        if Chem.MolFromSmiles(word) is not None:
          return word
    return None

  def predict_molecule(self, smiles: str) -> str:
    node_features, edge_index, valid = smiles_to_features(smiles)
    if not valid:
      return {'error': 'Invalid SMILES string.'}

    x = torch.tensor(node_features, dtype = torch.float)
    edge_index = torch.tensor(edge_index, dtype = torch.long).t().contiguous()
    batch = torch.zeros(x.shape[0], dtype = torch.long)

    data = Data(x=x, edge_index = edge_index, batch = batch).to(self.device)

    self.gnn.eval()
    with torch.no_grad():
      prediction = self.gnn(data).item()

    descriptors = get_basic_descriptors(smiles)

    return {
        'smiles': smiles,
        'homo_lumo_gap': round(prediction, 4),
        'unit': 'Hartree',
        'descriptors': descriptors
    }

  def ask_llm(self, question: str) -> str:
    if self.llm is None:
      return "LLM  not loaded."
    model, tokenizer = self.llm
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    prompt = f"""### Instruction:
  {question}

  ### Response:
  """
    inputs = tokenizer(prompt, return_tensors = 'pt').to('cuda')
    outputs = model.generate(
        **inputs,
        max_new_tokens=40,
        max_length=None,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens = True)
    return response.split('### Response:')[-1].strip()


  def run(self, user_input: str) -> dict:
    route = self.classify_input(user_input)

    if route == 'gnn':
      smiles = self.extract_smiles(user_input)
      if smiles is None:
        return {
            'route': 'gnn',
            'error': 'No valid SMILES found.',
            'suggestion': 'Example: CC(=O)O for acetic acid'

        }
      if self.gnn is None:
        return {'route': 'gnn', 'error': 'GNN model not loaded.'}
      result = self.predict_molecule(smiles)
      result['route'] = 'gnn'
      return result
    else:
      response = self.ask_llm(user_input)
      return {
          'route': 'llm',
          'question': user_input,
          'answer': response
      }
