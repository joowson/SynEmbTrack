from synembtrack.synth.composit.runner import run_once



if __name__ == "__main__":
    
    
    
    raw_data_code = "demo_2Dsuspension_25C" 
    
    # synth_img_code = "synthSet_VSreal"   
    synth_img_code = "synthSet_demoTrainset"   
    synth_img_code = "synthSet_demoCVset"   

    
    
    ### --- Run

    run_once(synth_img_code)