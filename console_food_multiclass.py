'''
Created on 31 mai 2016

@author: wangxin
'''
def main():
    import myio,myopt
    import os, time
    dataSource = "local"
     
    gazeType = "food";
    taskName = "chain_glsvm_nonlinear_jumping_50_1e-4/";

#     classes = [sys.argv[1]]
#     scaleCV = [int(sys.argv[2])]    
    scaleCV=[50]
    output_typ="multiclass"
    model_name = "tf_mlp_multiclass_clf"
    category_num = 20
    assert output_typ in model_name
    categories = [
            "apple-pie",
#             "bread-pudding",
#             "beef-carpaccio",
#             "beet-salad",
#             "chocolate-cake",
#             "chocolate-mousse",
#             "donuts",
#             "beignets",
#             "eggs-benedict",
#             "croque-madame",
#             "gnocchi",
#             "shrimp-and-grits",
#             "grilled-salmon",
#             "pork-chop",
#             "lasagna",
#             "ravioli",
#             "pancakes",
#             "french-toast",
#             "spaghetti-bolognese",
#             "pad-thai"        
            ]

#     categories = ["apple-pie"]

    if dataSource=="local":
        sourceDir = "/local/wangxin/Data/UPMC_Food_Gaze_20/";
        resDir = "/local/wangxin/results/upmc_food/";        
    elif dataSource=="big":
        sourceDir = "/home/wangxin/Data/UPMC_Food_Gaze_20/";
        resDir = "/home/wangxin/results/upmc_food/";
   

    resultFolder = resDir+taskName;
    
    resultFilePath = resultFolder + "ap_summary_ecarttype_seed1_detail.txt";
    metricFolder = resultFolder + "metric/";
    classifierFolder = resultFolder + "classifier/";
    scoreFolder = resultFolder + "score/";
    trainingDetailFolder = resultFolder + "trainingdetail/";
    featureFolder = sourceDir+"vgg-m-2048_features_new/"
    gazeFolder = sourceDir+"gazeloss_folder_noclasswise/"
    
    
    learning_rate = 0.0002;
    MAX_NB_ITER = 10000
    BATCH_SIZE = 3
    gamma = 0.2
    train_bool=True 
    if not train_bool:
        model_id="1474042773"

    print "Experiment detail:"
    print "categories \t%s"%str(categories),
    print "\tscale \t\t%s"%str(scaleCV)
    print "\tgamma \t\t%s"%str(gamma)
    
    for scale in scaleCV:
        for category in categories:
            if output_typ == "multiclass":
                example_file = "example_files_multi_class"
                train_example_list_root_folder = sourceDir + example_file+"/"+'_'.join(["train", str(scale), "matconvnet_m_2048_layer_20.txt"])
                valval_example_list_root_folder = sourceDir + example_file+"/"+'_'.join(["val", str(scale), "matconvnet_m_2048_layer_20.txt"])
                valtest_example_list_root_folder = sourceDir + example_file+"/"+'_'.join(["test", str(scale), "matconvnet_m_2048_layer_20.txt"])
            
            elif output_typ == "binaryclass":
                example_file = "example_files"
                train_example_list_root_folder = sourceDir + example_file+"/"+str(scale)+"/"+'_'.join([category,"train_scale", str(scale), "matconvnet_m_2048_layer_20.txt"])
                valval_example_list_root_folder = sourceDir + example_file+"/"+str(scale)+"/"+'_'.join([category,"val_scale", str(scale), "matconvnet_m_2048_layer_20.txt"])
                valtest_example_list_root_folder = sourceDir + example_file+"/"+str(scale)+"/"+'_'.join([category,"test_scale", str(scale), "matconvnet_m_2048_layer_20.txt"])
            
            nb_instance = myio.scale2RowNumber(scale)**2
            print "reading ok"
            #models, stefan|ferrari|food, category, scale,  
            SAVE_FOLDER = os.path.join("/local/wangxin/models",gazeType,model_name, str(scale))
            if not os.path.exists(SAVE_FOLDER):
                os.makedirs(SAVE_FOLDER)
            if train_bool:
                SAVE_PATH = SAVE_FOLDER+'/'+str(int(time.time()))+'.tfmodel'
            else:
                SAVE_PATH = SAVE_FOLDER+'/'+model_id+'.tfmodel'
            
            # title says all
            if model_name=="tf_mil_conv_detection":
                train_bags = myio.load_binary_bags(train_example_list_root_folder, scale, featureFolder, gazeFolder)
                valval_bags = myio.load_binary_bags(valval_example_list_root_folder, scale, featureFolder, gazeFolder)
                valtest_bags = myio.load_binary_bags(valtest_example_list_root_folder, scale, featureFolder, gazeFolder)
                
                X_train , y_train, Gazes_train = myio.organize_examples_by_bags(train_bags, nb_instance)
                X_test, y_test, Gazes_test = myio.organize_examples_by_bags(valtest_bags, nb_instance)
                myopt.tf_mil_conv_detection(X_train , y_train, Gazes_train, X_test, y_test, Gazes_test,
                               MAX_NB_ITER, BATCH_SIZE,gamma)
            # title says all
            if model_name == "tf_linear_gaze_reg":
                train_bags = myio.load_binary_bags(train_example_list_root_folder, scale, featureFolder, gazeFolder)
                valval_bags = myio.load_binary_bags(valval_example_list_root_folder, scale, featureFolder, gazeFolder)
                valtest_bags = myio.load_binary_bags(valtest_example_list_root_folder, scale, featureFolder, gazeFolder)
                
                X_train , y_train, Gazes_train = myio.organize_examples_by_bags(train_bags, nb_instance)
                X_test, y_test, Gazes_test = myio.organize_examples_by_bags(valtest_bags, nb_instance)
                myopt.tf_linear_gaze_reg(X_train , y_train, Gazes_train, X_test, y_test, Gazes_test,
                                learning_rate, MAX_NB_ITER,BATCH_SIZE,)
            # title says all
            
            # gaze is a scalar value
            # regression of gaze loss for patches of images of given category
            # classification is done by taking the maximum of the gaze density response
            if model_name == "tf_mlp_gaze_reg_classwise_clf_binary":
                if train_bool:
                    train_bags = myio.load_binary_bags(train_example_list_root_folder, scale, featureFolder, gazeFolder)
                    valval_bags = myio.load_binary_bags(valval_example_list_root_folder, scale, featureFolder, gazeFolder)
                    valtest_bags = myio.load_binary_bags(valtest_example_list_root_folder, scale, featureFolder, gazeFolder)
                    X_train, y_train, Gazes_train = myio.organize_examples_by_instances_classwise(train_bags, nb_instance)
                    X_val, y_val, Gazes_val = myio.organize_examples_by_instances_classwise(valval_bags, nb_instance)
                    X_test, y_test, Gazes_test = myio.organize_examples_by_binaryclass_bags(valtest_bags,nb_instance)
                    myopt.tf_mlp_gaze_reg_singleclass(X_train , y_train, Gazes_train,
                                          X_val, y_val, Gazes_val, 
                                          X_test, y_test, Gazes_test,
                                          SAVE_PATH, MAX_NB_ITER, BATCH_SIZE,train_bool, category,scale)
                else:
                    valtest_bags = myio.load_binary_bags(valtest_example_list_root_folder, scale, featureFolder, gazeFolder, stochastic=False)
                    X_test, y_test, Gazes_test = myio.organize_examples_by_binaryclass_bags(valtest_bags,nb_instance)

                    myopt.tf_mlp_gaze_reg_singleclass(None,None,None,
                                          None,None,None, 
                                          X_test, y_test, Gazes_test,
                                          SAVE_PATH, MAX_NB_ITER, BATCH_SIZE,train_bool, category,scale)    
            if output_typ == "binaryclass":
                train_bags = myio.load_binary_bags(train_example_list_root_folder, scale, featureFolder, gazeFolder)
                valval_bags = myio.load_binary_bags(valval_example_list_root_folder, scale, featureFolder, gazeFolder)
                valtest_bags = myio.load_binary_bags(valtest_example_list_root_folder, scale, featureFolder, gazeFolder)
                X_train, y_train, Gazes_train = myio.organize_examples_by_instances(train_bags,  nb_instance)
                X_val, y_val, Gazes_val = myio.organize_examples_by_instances(valval_bags, nb_instance)
                X_test, y_test, Gazes_test = myio.organize_examples_by_bags(valtest_bags, nb_instance)
                if model_name == "tf_mlp_gaze_reg_universal_clf_binary":
                    """ gaze is a scalar value     
                        regression of gaze loss for all patches of all images from any category            
                        classification is followed by training a SVM"""
                    
                    myopt.tf_mlp_gaze_reg_universal(X_train , y_train, Gazes_train,
                                      X_val, y_val, Gazes_val, 
                                      X_test, y_test, Gazes_test,
                                      SAVE_PATH, MAX_NB_ITER, BATCH_SIZE,train_bool, category)

            if output_typ == "multiclass":
                print "mode:%s"%output_typ
                train_bags = myio.load_multiclass_bags(train_example_list_root_folder, scale, featureFolder, gazeFolder)
                valval_bags = myio.load_multiclass_bags(valval_example_list_root_folder, scale, featureFolder, gazeFolder)
                valtest_bags = myio.load_multiclass_bags(valtest_example_list_root_folder, scale, featureFolder, gazeFolder)

                X_train, y_train, Gazes_train = myio.organize_examples_by_instances_gaze_multiclass(train_bags, category_num, nb_instance)
                X_val, y_val, Gazes_val = myio.organize_examples_by_instances_gaze_multiclass(valval_bags, category_num, nb_instance)
                X_test, y_test, Gazes_test = myio.organize_examples_by_multiclass_bags(valtest_bags, category_num, nb_instance)
                # gaze is a one-hot vector, training in a multiclass mode. Label is integrated into gaze vector  
                if model_name == "tf_mlp_gaze_regclf_multiclass":
                    myopt.tf_mlp_gaze_regclf_multiclass(X_train , y_train, Gazes_train,
                                      X_val, y_val, Gazes_val, 
                                      X_test, y_test, Gazes_test,
                                      SAVE_PATH, MAX_NB_ITER, BATCH_SIZE,train_bool, category, category_num)
                # label is same for all regions
                if model_name == "tf_mlp_multiclass_clf":
                    myopt.tf_mlp_multiclass_clf(X_train , y_train, Gazes_train,
                                      X_val, y_val, Gazes_val, 
                                      X_test, y_test, Gazes_test,
                                      SAVE_PATH, MAX_NB_ITER, BATCH_SIZE,train_bool, category, category_num)
                if model_name == "multiclass_svm":
                    myopt.multiclass_svm(X_train , y_train, Gazes_train,
                                      X_val, y_val, Gazes_val, 
                                      X_test, y_test, Gazes_test,)
if __name__ == "__main__":
    main()