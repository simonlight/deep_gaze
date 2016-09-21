
def main():
    import myio,myopt
    import os, time
    dataSource = "local"
    taskName = "test_chain_glsvm_nonlinear_jumping_50_1e-4/";
    gazeType = "stefan"
#     classes = [sys.argv[1]]
#     scaleCV = [int(sys.argv[2])]
    # there is only one salience mode lfor stefan dataset
#     categories = ["jumping", "phoning", "playinginstrument", "reading", "ridingbike", "ridinghorse", "running", "takingphoto", "usingcomputer", "walking"]
#     categories = ["usingcomputer", "walking"]
    scaleCV=[50]
    model_name = "mlp_category_reg_clf"

    if dataSource=="local":
        sourceDir = "/local/wangxin/Data/full_stefan_gaze/"
        resDir = "/local/wangxin/results/full_stefan_gaze/%s/"%model_name        
    elif dataSource=="big":
        sourceDir = "/net/urbandancesquad/wangxin/Data/full_stefan_gaze/"
        resDir = "/net/urbandancesquad/wangxin/results/full_stefan_gaze/chain_glsvm_et/"

    resultFolder = resDir+taskName;
    
    resultFilePath = resultFolder + "ap_summary_ecarttype_seed1_detail.txt"
    metricFolder = resultFolder + "metric/"
    classifierFolder = resultFolder + "classifier/"
    scoreFolder = resultFolder + "score/"
    trainingDetailFolder = resultFolder + "trainingdetail/"
    featureFolder = sourceDir+"m2048_trainval_features/"
    gazeFolder = sourceDir+"gazeloss_files/"
    
    saveClassifier = True;
    loadClassifier = False;
    
    learning_rate=0.0002;
    MAX_NB_ITER=1000
    BATCH_SIZE=128
    gamma=0.2
    train_bool=True
    if not train_bool:
        model_id="???"
    
    print "Experiment detail:"
    print "categories \t%s"%str(categories),
    print "\tscale \t\t%s"%str(scaleCV)
    print "\tgamma \t\t%s"%str(gamma)
    
    for category in categories:
        for scale in scaleCV:
#             full_example_list_root_folder = sourceDir + "example_files/"+str(100)+"/" + '_'.join([category, "train", "scale", str(100), "matconvnet_m_2048_layer_20.txt"])
#             full_image_bags = myio.load_bags(full_example_list_root_folder, scale, featureFolder, gazeFolder)
            train_example_list_root_folder = sourceDir + "example_files_with_gaze_annotation/"+str(scale)+"/" + '_'.join([category, "train", "scale", str(scale), "matconvnet_m_2048_layer_20.txt"])
            valval_example_list_root_folder = sourceDir + "example_files_with_gaze_annotation/"+str(scale)+"/" + '_'.join([category, "valval", "scale", str(scale), "matconvnet_m_2048_layer_20.txt"])
            valtest_example_list_root_folder = sourceDir + "example_files_with_gaze_annotation/"+str(scale)+"/" + '_'.join([category, "valtest", "scale", str(scale), "matconvnet_m_2048_layer_20.txt"])
            
            train_bags = myio.load_bags(train_example_list_root_folder, scale, featureFolder, gazeFolder)
            valval_bags = myio.load_bags(valval_example_list_root_folder, scale, featureFolder, gazeFolder)
            valtest_bags = myio.load_bags(valtest_example_list_root_folder, scale, featureFolder, gazeFolder)
            
            nb_instance = myio.scale2RowNumber(scale)**2
            
            #models, stefan|ferrari|food, category, scale,  
            SAVE_FOLDER = os.path.join("models",gazeType,model_name, str(scale))
            if not os.path.exists(SAVE_FOLDER):
                os.makedirs(SAVE_FOLDER)
            if train_bool:
                SAVE_PATH = SAVE_FOLDER+'/'+str(int(time.time()))+'.tfmodel'
            else:
                SAVE_PATH = SAVE_FOLDER+'/'+model_id+'.tfmodel'
                        
            if model_name=="deep_laptev":
                X_train , y_train, Gazes_train = myio.organize_examples_by_bags(train_bags, nb_instance)
                X_test, y_test, Gazes_test = myio.organize_examples_by_bags(valtest_bags, nb_instance)
                myopt.tf_mil_conv_detection(X_train , y_train, Gazes_train, X_test, y_test, Gazes_test,
                               MAX_NB_ITER, BATCH_SIZE,gamma)
            if model_name == "lin_reg":
                X_train , y_train, Gazes_train = myio.organize_examples_by_bags(train_bags, nb_instance)
                X_test, y_test, Gazes_test = myio.organize_examples_by_bags(valtest_bags, nb_instance)
                myopt.tf_linear_gaze_reg(X_train , y_train, Gazes_train, X_test, y_test, Gazes_test,
                                learning_rate, MAX_NB_ITER,BATCH_SIZE,)
            if model_name == "mlp_reg":
                X_train , y_train, Gazes_train = myio.organize_examples_by_instances(train_bags, nb_instance)
                X_test, y_test, Gazes_test = myio.organize_examples_by_instances(valtest_bags, nb_instance)
                myopt.tf_mlp_gaze_reg(X_train , y_train, Gazes_train, X_test, y_test, Gazes_test,
                                learning_rate, MAX_NB_ITER,BATCH_SIZE,)
            if model_name == "full_image_svm":
                X_train , y_train, Gazes_train = myio.organize_examples_by_instances(train_bags, nb_instance)
                X_test, y_test, Gazes_test = myio.organize_examples_by_instances(valtest_bags, nb_instance)
                myopt.sk_fullimage_svm_clf(X_train , y_train, X_test, y_test)
            if model_name == "mlp_reg_clf":
                X_train, y_train, Gazes_train = myio.organize_examples_by_instances(train_bags, nb_instance)
                X_val, y_val, Gazes_val = myio.organize_examples_by_instances(valval_bags, nb_instance)
                X_test, y_test, Gazes_test = myio.organize_examples_by_bags(valtest_bags, nb_instance)
                myopt.tf_mlp_gaze_reg(X_train , y_train, Gazes_train,
                                      X_val, y_val, Gazes_val, 
                                      X_test, y_test, Gazes_test,
                                      SAVE_PATH, MAX_NB_ITER, BATCH_SIZE, train_bool, category)
            if model_name == "mlp_category_reg_clf":
                X_train, y_train, Gazes_train = myio.organize_examples_by_instances_category(train_bags, nb_instance)
                X_val, y_val, Gazes_val = myio.organize_examples_by_instances_category(valval_bags, nb_instance)
                X_test, y_test, Gazes_test = myio.organize_examples_by_bags(valtest_bags, nb_instance)
                myopt.tf_mlp_gaze_reg(X_train , y_train, Gazes_train,
                                      X_val, y_val, Gazes_val, 
                                      X_test, y_test, Gazes_test,
                                      SAVE_PATH, MAX_NB_ITER, BATCH_SIZE,train_bool, category)                
if __name__ == "__main__":
    main()
