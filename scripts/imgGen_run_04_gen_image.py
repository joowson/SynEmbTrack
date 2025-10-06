from synembtrack.synth.composit.runner import run_once



if __name__ == "__main__":



    raw_data_code = "demo_2Dsuspension_25C"

    # synth_img_code = "synthSet_VSreal"


    # for training data
    synth_img_code = "synthSet_demoTrain"

    # for validation data
    synth_img_code = "synthSet_demoVal"



    ### --- Run

    run_once(synth_img_code)
