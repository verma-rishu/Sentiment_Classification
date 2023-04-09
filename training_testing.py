import torch
import time
import pandas as pd
import torch.optim as opt
from dataset import *
from cnn_model import *
from sklearn.metrics import classification_report

def main():
    # Data file and word embedding model
    data_file_path = './yadr_chunks.csv'
    top_data_df = pd.read_csv(data_file_path)
    print("Columns in the original dataset:\n")
    print(top_data_df.columns)
    print("Number of rows per star rating:")
    print(top_data_df['stars'].value_counts())


    # Mapping stars to sentiment into three categories
    top_data_df['sentiment'] = [ map_sentiment(x) for x in top_data_df['stars']]
    # Plotting the sentiment distribution
    plt.figure()
    pd.value_counts(top_data_df['sentiment']).plot.bar(title="Sentiment distribution in df")
    plt.xlabel("Sentiment")
    plt.ylabel("No. of rows in df")
    plt.show()

    # Function call to get the top 10000 from each sentiment
    top_data_df_small = get_top_data(top_data_df, top_n=10000)

    # After selecting top few samples of each sentiment
    print("After segregating and taking equal number of rows for each sentiment:")
    print(top_data_df_small['sentiment'].value_counts())
    print(top_data_df_small.head(10))

    # Tokenize the text column to get the new column 'tokenized_text'
    top_data_df_small['tokenized_text'] = [simple_preprocess(line, deacc=True) for line in top_data_df_small['text']] 
    print(top_data_df_small['tokenized_text'].head(10))

    #STEMMING
    porter_stemmer = PorterStemmer()

    # Get the stemmed_tokens
    top_data_df_small['stemmed_tokens'] = [[porter_stemmer.stem(word) for word in tokens] for tokens in top_data_df_small['tokenized_text'] ]
    print(top_data_df_small['stemmed_tokens'].head(10))

    # Call the train_test_split
    X_train, X_test, Y_train, Y_test = split_train_test(top_data_df_small)
    
    # Device initialization
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    #Genreating word embeddings
    size = 500
    window = 3
    min_count = 1
    workers = 3
    sg = 1

    # Train Word2vec model
    w2vmodel, word2vec_file = make_word2vec_model(top_data_df_small, padding=True, sg=sg, min_count=min_count, size=size, workers=workers, window=window)
    w2vmodel.save(word2vec_file)
    max_sen_len = top_data_df_small.stemmed_tokens.map(len).max()
    padding_idx = w2vmodel.wv.key_to_index["pad"] 


    NUM_CLASSES = 3 #positive, negative, neutral
    VOCAB_SIZE = len(w2vmodel.wv)

    cnn_model = CnnTextClassifier(vocab_size=VOCAB_SIZE, num_classes=NUM_CLASSES)
    cnn_model.to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = opt.Adam(cnn_model.parameters(), lr=0.001)
    num_epochs = 30

    # Open the file for writing loss
    loss_file_name = 'cnn_class_big_loss_with_padding.csv'
    f = open(loss_file_name,'w')
    f.write('iter, loss')
    f.write('\n')

    cnn_model.train()
    for epoch in range(num_epochs):
        start_time = time.time()
        print("Epoch" + str(epoch + 1))
        train_loss = 0
        for index, row in X_train.iterrows():
            # Clearing the accumulated gradients
            cnn_model.zero_grad()

            # Make the bag of words vector for stemmed tokens 
            bow_vec = make_word2vec_vector_cnn(row['stemmed_tokens'], max_sen_len, padding_idx, w2vmodel, device)
        
            # Forward pass to get output
            probs = cnn_model(bow_vec)

            # Get the target label
            target = make_target(Y_train['sentiment'][index], device)

            # Calculate Loss: softmax --> cross entropy loss
            loss = loss_function(probs, target)
            train_loss += loss.item()

            # Getting gradients w.r.t. parameters
            loss.backward()

            # Updating parameters
            optimizer.step()


        # if index == 0:
        #     continue
        print(f"Epoch ran : {str(epoch+1)} Time Taken : {str(time.time() - start_time)}")
        f.write(str((epoch+1)) + "," + str(train_loss / len(X_train)))
        f.write('\n')
        train_loss = 0

    torch.save(cnn_model, 'cnn_big_model_500_with_padding.pth')

    f.close()
    print("Input vector")
    print(bow_vec.cpu().numpy())
    print("Probs")
    print(probs)
    print(torch.argmax(probs, dim=1).cpu().numpy()[0])
        

    bow_cnn_predictions = []
    original_lables_cnn_bow = []

    cnn_model.eval()

    loss_df = pd.read_csv('cnn_class_big_loss_with_padding.csv')
    print(loss_df.columns)
    loss_df.plot(' loss')
    with torch.no_grad():
        for index, row in X_test.iterrows():
            bow_vec = make_word2vec_vector_cnn(row['stemmed_tokens'],max_sen_len, padding_idx, w2vmodel, device)
            probs = cnn_model(bow_vec)
            _, predicted = torch.max(probs.data, 1)
            bow_cnn_predictions.append(predicted.cpu().numpy()[0])
            original_lables_cnn_bow.append(make_target(Y_test['sentiment'][index],device).cpu().numpy()[0])

    print(classification_report(original_lables_cnn_bow,bow_cnn_predictions))
    loss_file_name = 'cnn_class_big_loss_with_padding.csv'
    loss_df = pd.read_csv(loss_file_name)
    print(loss_df.columns)
    plt_500_padding_30_epochs = loss_df[' loss'].plot()
    fig = plt_500_padding_30_epochs.get_figure()
    fig.savefig('loss_plt_500_padding_30_epochs.pdf')

if __name__=='__main__':
    main()
