from rdkit import Chem
from leadopt.config.preset_loader import PresetLoader

lead = "N#Cc1cc(-n2cc(C(=O)O)nn2)ccc1OCc1ccc(Cl)cc1"
loaded = PresetLoader().load("F:/Phuong/Project/leadopt/examples/1. QSAR/preset_qsar.yaml")
mol = Chem.MolFromSmiles(lead)
res = loaded.scorer.score(mol)

print("valid:", res.valid)
print("objective:", res.objective)
print("fail_reason:", res.fail_reason)
print("components:", res.components)
print("metadata:", res.metadata)