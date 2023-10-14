% Importar los datos desde el archivo CSV
data = csvread('irisbin.csv');
X = data(:, 1:4);  % Características (primeras cuatro columnas)
Y = data(:, 5:7);  % Etiquetas (últimas tres columnas)

% Definir la arquitectura de la red neuronal
hidden_layers = [100, 100, 600];
regularization = 0.01;
learning_rate = 0.01;

% Inicializar variables para almacenar la precisión de Leave-k-Out y Leave-One-Out
accuracy_leave_k_out = 0;
accuracy_leave_one_out = 0;

% Valores de k para Leave-k-Out
k_values = [5, 10];

% Bucle para Leave-k-Out
for k = k_values
    % Crear particiones Leave-k-Out
    cv_k_out = cvpartition(size(X, 1), 'KFold', k);

    % Bucle para las particiones de Leave-k-Out
    for fold = 1:cv_k_out.NumTestSets
        train_idx = training(cv_k_out, fold);
        test_idx = test(cv_k_out, fold);

        X_train_k_out = X(train_idx, :);
        Y_train_k_out = Y(train_idx, :);
        X_test_k_out = X(test_idx, :);
        Y_test_k_out = Y(test_idx, :);

        % Crear un perceptrón multicapa
        net = patternnet(hidden_layers);
        net.performParam.regularization = regularization;
        net.trainParam.lr = learning_rate;

        % Entrenar la red neuronal
        net = train(net, X_train_k_out', Y_train_k_out');

        % Hacer predicciones en los datos de prueba
        Y_pred_k_out = net(X_test_k_out');

        % Convertir las salidas a etiquetas (Setosa, Versicolor, Virginica)
        labels_pred_k_out = round(Y_pred_k_out)';

        % Calcular la precisión y agregarla al acumulador
        accuracy_leave_k_out = accuracy_leave_k_out + sum(all(labels_pred_k_out == Y_test_k_out, 2)) / size(Y_test_k_out, 1);
    end

    % Calcular la precisión promedio para Leave-k-Out
    accuracy_leave_k_out = accuracy_leave_k_out / cv_k_out.NumTestSets;
    disp(['Precisión Leave-', num2str(k), '-Out: ', num2str(accuracy_leave_k_out * 100), '%']);
end

% Crear particiones Leave-One-Out
cv_leave_one_out = cvpartition(size(X, 1), 'LeaveOut');

% Bucle para las particiones Leave-One-Out
for fold = 1:cv_leave_one_out.NumTestSets
    train_idx = training(cv_leave_one_out, fold);
    test_idx = test(cv_leave_one_out, fold);

    X_train_one_out = X(train_idx, :);
    Y_train_one_out = Y(train_idx, :);
    X_test_one_out = X(test_idx, :);
    Y_test_one_out = Y(test_idx, :);

    % Crear un perceptrón multicapa
    net = patternnet(hidden_layers);
    net.performParam.regularization = regularization;
    net.trainParam.lr = learning_rate;

    % Entrenar la red neuronal
    net = train(net, X_train_one_out', Y_train_one_out');

    % Hacer predicciones en los datos de prueba
    Y_pred_one_out = net(X_test_one_out');

    % Convertir las salidas a etiquetas (Setosa, Versicolor, Virginica)
    labels_pred_one_out = round(Y_pred_one_out)';

    % Calcular la precisión y agregarla al acumulador
    accuracy_leave_one_out = accuracy_leave_one_out + sum(all(labels_pred_one_out == Y_test_one_out, 2)) / size(Y_test_one_out, 1);
end

% Calcular la precisión promedio para Leave-One-Out
accuracy_leave_one_out = accuracy_leave_one_out / cv_leave_one_out.NumTestSets;
disp(['Precisión Leave-One-Out: ', num2str(accuracy_leave_one_out * 100), '%']);
