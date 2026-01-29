#include <bits/stdc++.h>
#include <fstream>
#include <sstream>
#include <random>
#include <cstdlib>
#include <ctime>
#include <cctype>
#include <iomanip>

using namespace std;

struct CommentData {
    string text;
    vector<int> labels;
};
double sigmoid(double z) {
    if (z > 20.0) return 0.9999999;
    if (z < -20.0) return 0.0000001;
    return 1.0 / (1.0 + exp(-z));
}

vector<CommentData> parseToxicCSV(const string& filename) {
    ifstream file(filename);
    vector<CommentData> comments;
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        return comments;
    }

    string line;
    if (!getline(file, line)) return comments;

    auto countQuotes = [](const string& s) {
        return count(s.begin(), s.end(), '\"');
    };

    while (getline(file, line)) {
        if (line.empty()) continue;
        string record = line;
        if (!record.empty() && record.back() == '\r') record.pop_back();

        while (countQuotes(record) % 2 != 0) {
            if (!getline(file, line)) break;
            if (!line.empty() && line.back() == '\r') line.pop_back();
            record += "\n" + line;
        }

        CommentData data;
        size_t pos = record.find(',');
        if (pos == string::npos) continue;

        size_t start = pos + 1;
        string text;

        if (start < record.size() && record[start] == '\"') {
            size_t i = start + 1;
            while (i < record.size()) {
                if (record[i] == '\"') {
                    if (i + 1 < record.size() && record[i+1] == '\"') {
                        text.push_back('\"');
                        i += 2;
                    } else {
                        i++;
                        break;
                    }
                } else {
                    text.push_back(record[i]);
                    i++;
                }
            }
            if (i < record.size() && record[i] == ',') i++;
            start = i;
        } else {
            size_t end = record.find(',', start);
            text = record.substr(start, end - start);
            if (end == string::npos) continue;
            start = end + 1;
        }

        data.text = text;
        data.labels.reserve(6);

        for (int lbl = 0; lbl < 6; ++lbl) {
            size_t comma = record.find(',', start);
            string field;
            if (lbl < 5) {
                if (comma == string::npos) {
                    field = record.substr(start);
                    start = record.size();
                } else {
                    field = record.substr(start, comma - start);
                    start = comma + 1;
                }
            } else {
                field = record.substr(start);
            }

            try {
                int val = stoi(field);
                data.labels.push_back(val > 0 ? 1 : 0);
            } catch (...) {
                data.labels.push_back(0);
            }
        }
        comments.push_back(move(data));
    }
    return comments;
}

vector<string> tokenize(const string &text, const unordered_set<string> &stopwords) {
    string s;
    for(char c : text) {
        if (isalpha((unsigned char)c) || isspace((unsigned char)c))
            s += tolower((unsigned char)c);
        else
            s += ' ';
    }
    vector<string> tokens;
    istringstream iss(s);
    while (iss >> s) {
        if (!stopwords.count(s) && s.length() > 1) tokens.push_back(s);
    }
    return tokens;
}

unordered_map<string,int> buildVocab(const vector<vector<string>> &docs, int topN) {
    unordered_map<string,int> freq;
    for (auto &doc: docs) {
        for (auto &w: doc) freq[w]++;
    }
    vector<pair<int,string>> items;
    items.reserve(freq.size());
    for (auto &p: freq) items.push_back({p.second, p.first});

    sort(items.begin(), items.end(), [](auto &a, auto &b){ return a.first > b.first; });

    unordered_map<string,int> vocab;
    vocab.reserve(topN);
    int idx = 0;
    for (auto &p: items) {
        if (idx >= topN) break;
        vocab[p.second] = idx++;
    }
    return vocab;
}

void computeTFIDF(const vector<vector<string>> &docs,
                  const unordered_map<string,int> &vocab,
                  vector<double> &idf,
                  vector<vector<pair<int,double>>> &nonzero) {

    int N = docs.size();
    int V = vocab.size();
    if (V == 0) return;

    vector<int> df(V, 0);
    for (auto &doc: docs) {
        unordered_set<int> seen;
        for (auto &w: doc) {
            auto it = vocab.find(w);
            if (it != vocab.end()) seen.insert(it->second);
        }
        for (int id: seen) df[id]++;
    }

    idf.resize(V);
    for (auto &p: vocab) {
        int id = p.second;
        idf[id] = log((1.0 + N) / (1.0 + df[id])) + 1.0;
    }

    nonzero.resize(N);
    for (int i = 0; i < N; i++) {
        unordered_map<int,int> counts;
        for (auto &w: docs[i]) {
            auto it = vocab.find(w);
            if (it != vocab.end()) counts[it->second]++;
        }

        double norm = 0.0;
        vector<pair<int,double>> temp_doc;
        for (auto &p: counts) {
            int id = p.first;
            double tf = 1.0 + log(p.second);
            double tfidf = tf * idf[id];
            if(tfidf > 1e-9) {
                temp_doc.push_back({id, tfidf});
                norm += tfidf * tfidf;
            }
        }

        norm = sqrt(norm);
        if (norm > 1e-9) {
            for (auto &p : temp_doc) {
                p.second /= norm;
                nonzero[i].push_back(p);
            }
        }
    }
}

int main() {
    random_device rd;
    mt19937 g(rd());
    srand(42);

    unordered_set<string> stopwords = {"the","and","is","in","to","of","a","that","it","on","for","with"};

    cout << "Reading train.csv..." << endl;
    vector<CommentData> data = parseToxicCSV("train.csv");
    if(data.empty()) {
        cout << "Error: No data loaded. Check if train.csv exists." << endl;
        return 0;
    }
    shuffle(data.begin(), data.end(), g);

    vector<CommentData> data2;
    int clean_cnt = 0;
    for(const auto& x : data) {
        int label_sum = 0;
        for(int l : x.labels) label_sum += l;

        if(label_sum == 0) {
            clean_cnt++;
            if(clean_cnt < 30000) data2.push_back(x);
        } else {
            // Upsample toxic
            if(x.labels[1]||x.labels[3]||x.labels[5]) { for(int i=0; i<5; i++) data2.push_back(x); }
            if(x.labels[3]) { for(int i=0; i<2; i++) data2.push_back(x); }
            data2.push_back(x);
        }
    }
    shuffle(data2.begin(), data2.end(), g);

    cout << "Training on " << data2.size() << " comments." << endl;

    vector<string> raw_comments;
    vector<vector<int>> labels;
    for(const auto& x : data2) {
        raw_comments.push_back(x.text);
        labels.push_back(x.labels);
    }

    vector<vector<string>> docs;
    for (auto &text : raw_comments) {
        docs.push_back(tokenize(text, stopwords));
    }

    int TOP_N = 200000;
    auto vocab = buildVocab(docs, TOP_N);
    int V = vocab.size();
    int N_docs = docs.size();
    cout << "Vocab size: " << V << endl;
    vector<double> idf;
    vector<vector<pair<int,double>>> nonzero(N_docs);
    computeTFIDF(docs, vocab, idf, nonzero);
    int hiddenSize = 64, outputSize = 6;
    double lr = 0.01;
    vector<vector<double>> W1(hiddenSize, vector<double>(V));
    vector<double> b1(hiddenSize, 0.0);
    vector<vector<double>> W2(outputSize, vector<double>(hiddenSize));
    vector<double> b2(outputSize, 0.0);
    for (int i = 0; i < hiddenSize; i++)
        for (int j = 0; j < V; j++)
            W1[i][j] = ((double)rand()/RAND_MAX - 0.5) * 0.01;

    for (int i = 0; i < outputSize; i++)
        for (int j = 0; j < hiddenSize; j++)
            W2[i][j] = ((double)rand()/RAND_MAX - 0.5) * 0.01;

    int epochs = 5;
    int training_size = (N_docs * 9) / 10;

    for (int epoch = 0; epoch < epochs; epoch++) {
        double epoch_loss = 0.0;
        int nan_count = 0;

        for (int i = 0; i < training_size; i++) {
            vector<double> hidden(hiddenSize);
            for (int h = 0; h < hiddenSize; h++) {
                double z = b1[h];
                for (auto &p : nonzero[i]) z += W1[h][p.first] * p.second;
                hidden[h] = max(0.0, z); // ReLU
            }

            vector<double> out(outputSize);
            for (int k = 0; k < outputSize; k++) {
                double z = b2[k];
                for (int h = 0; h < hiddenSize; h++) z += W2[k][h] * hidden[h];
                out[k] = sigmoid(z);
            }

            vector<double> delta_out(outputSize);
            for (int k = 0; k < outputSize; k++) {
                double y = (double)labels[i][k];
                double pred = out[k];
                if(pred < 1e-7) pred = 1e-7;
                if(pred > 1.0 - 1e-7) pred = 1.0 - 1e-7;

                epoch_loss -= y * log(pred) + (1.0 - y) * log(1.0 - pred);
                delta_out[k] = out[k] - y;
            }

            vector<double> delta_hidden(hiddenSize, 0.0);
            for (int h = 0; h < hiddenSize; h++) {
                double grad = 0.0;
                for (int k = 0; k < outputSize; k++) grad += W2[k][h] * delta_out[k];
                delta_hidden[h] = (hidden[h] > 0) ? grad : 0.0;
            }

            for (int k = 0; k < outputSize; k++) {
                for (int h = 0; h < hiddenSize; h++) W2[k][h] -= lr * (delta_out[k] * hidden[h]);
                b2[k] -= lr * delta_out[k];
            }

            for (int h = 0; h < hiddenSize; h++) {
                for (auto &p : nonzero[i]) {
                    W1[h][p.first] -= lr * (delta_hidden[h] * p.second);
                }
                b1[h] -= lr * delta_hidden[h];
            }
        }

        double avg_loss = epoch_loss / training_size;
        cout << "Epoch " << epoch+1 << " - Loss: " << avg_loss << endl;

        if (isnan(avg_loss)) {
            cout << "CRITICAL FAILURE: Loss is NaN. Restarting weights..." << endl;
             for (int i = 0; i < hiddenSize; i++)
                for (int j = 0; j < V; j++) W1[i][j] = ((double)rand()/RAND_MAX - 0.5) * 0.001;
        }
    }

    cout << "\n--- Ready for Input ---\n";
    string line;
    vector<string> classNames = {"Toxic","Severe","Obscene","Threat","Insult","Hate"};

    while (true) {
        cout << "\nEnter comment: ";
        if (!getline(cin, line) || line.empty()) break;

        auto tokens = tokenize(line, stopwords);

        unordered_map<int,int> cnt;
        for (auto &w: tokens) {
            if(vocab.count(w)) cnt[vocab[w]]++;
        }

        vector<pair<int,double>> input_vec;
        double norm = 0.0;
        for(auto &p : cnt) {
            double tf = 1.0 + log(p.second);
            double val = tf * idf[p.first];
            input_vec.push_back({p.first, val});
            norm += val * val;
        }
        norm = sqrt(norm);
        if(norm > 0) for(auto &p : input_vec) p.second /= norm;

        vector<double> hidden(hiddenSize);
        for (int h = 0; h < hiddenSize; h++) {
            double z = b1[h];
            for (auto &p : input_vec) z += W1[h][p.first] * p.second;
            hidden[h] = max(0.0, z);
        }

        cout << "Result: ";
        for (int k = 0; k < outputSize; k++) {
            double z = b2[k];
            for (int h = 0; h < hiddenSize; h++) z += W2[k][h] * hidden[h];
            double p = sigmoid(z);
            if(p > 0.5) cout << "[" << classNames[k] << ": " << (int)(p*100) << "%] ";
        }
        cout << endl;
    }

    return 0;
}
