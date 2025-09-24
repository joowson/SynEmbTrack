
# run_pipeline.py
from synembtrack.cell_assoc.pipeline import main

if __name__ == "__main__":
    
    
    
    
    assoc_preset_name = 'demoAssoc'
    
    
    
    
    out_csv = main(assoc_key=assoc_preset_name)
    
    
    print(f"[OK] Saved → {out_csv}")
