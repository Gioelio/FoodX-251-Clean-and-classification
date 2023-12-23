def clean_dataset(aggregated, image_features, neighbors=200):
    count = 0
    from sklearn.neighbors import KNeighborsClassifier
    import tqdm
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(image_features['features'].values, image_features['label'].values)

    for i, row in tqdm.tqdm(image_features.iterrows()):
        pred = classifier.predict(row['features'])
        if pred != row['label']:
            count += 1
            print(count)
