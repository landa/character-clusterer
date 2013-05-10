all:
	g++ word_clusters.cpp -o word_clusters `pkg-config --libs opencv`
