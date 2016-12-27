library(h2o)

h2o.init()

train <- h2o.importFile('input/train.csv')
train$target <- log(train$SalePrice)

X <- setdiff(colnames(train), c('Id','SalePrice', 'target'))
y <- 'target'

model <- h2o.deeplearning(X, y, training_frame = train, model_id = 'dl_01', 
                         overwrite_with_best_model = TRUE,
                         use_all_factor_levels = TRUE, 
                         standardize = TRUE, activation = "RectifierWithDropout",
                         hidden = c(200,200), epochs = 10, variable_importances = TRUE, fold_assignment = "AUTO",
                         train_samples_per_iteration = -2, adaptive_rate = TRUE, input_dropout_ratio = 0.1,
                         l1 = 0.0001, l2 = 0, loss = "Huber", distribution = "AUTO", huber_alpha = 0.9,
                         score_interval = 5, score_training_samples = 10000, score_duty_cycle = 0.1,
                         stopping_rounds = 5, stopping_metric = "MSE", stopping_tolerance = 0.001, 
                         categorical_encoding = "AUTO", target_ratio_comm_to_comp = 0.05, seed=1234,
                         rho = 0.99, epsilon = 1e-8, nesterov_accelerated_gradient = TRUE,
                         initial_weight_distribution = "UniformAdaptive", nfolds = 10,
                         regression_stop = 0.000001, diagnostics = TRUE, fast_mode = TRUE,
                         force_load_balance = TRUE, single_node_mode = FALSE, shuffle_training_data = FALSE,
                         missing_values_handling = "MeanImputation", quiet_mode = FALSE,
                         sparse = FALSE, col_major = FALSE, sparsity_beta = 0,
                         max_categorical_features = 2147483647, reproducible = FALSE, export_weights_and_biases = FALSE)


h2o.rmse(model, xval = TRUE) #0.1292479

