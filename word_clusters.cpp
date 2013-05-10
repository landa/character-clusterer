#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>

cv::Scalar rc() { // get a random pastel color
  return cv::Scalar(150 + (float)rand()/((float)RAND_MAX/(100)),
                    150 + (float)rand()/((float)RAND_MAX/(100)),
                    150 + (float)rand()/((float)RAND_MAX/(100)));
}

const int UNASSIGNED_GROUP = 0;

const cv::Scalar GREY(100, 100, 100);
const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);
const cv::Scalar YELLOW(0, 220, 255);
const cv::Scalar CYAN(255, 255, 0);
const cv::Scalar MAGENTA(255, 0, 255);
const cv::Scalar BLACK(0, 0, 0);
const cv::Scalar COLORS[] = { GREY, rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(),
rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(),
rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc() };

// Clustering parameters
int MAX_DISSIMILARITY = 75.0;
const int MAX_ITERATIONS = 1000;
const int VERTICAL_SENSITIVITY = 2;

const char* WINDOW_NAME = "Clustering Characters";

typedef struct {
  cv::Rect rect;
  char value;
  int group;
} Character;

typedef struct {
  std::vector<Character> characters;
} Word;

cv::Mat canvas(800, 1000, CV_8UC3);

std::vector<Character> getCharacters(char* str, int offset_x, int offset_y) {
  std::vector<Character> characters;

  for (size_t ii = 0; ii < strlen(str); ++ii) {
    Character cur;
    cur.rect = cv::Rect(offset_x + ii*35, offset_y, 30, 40);
    cur.value = str[ii];
    cur.group = UNASSIGNED_GROUP;
    characters.push_back(cur);
  }

  return characters;
}

const char* getWordString(Word word) {
  std::stringstream ss;
  for (size_t ii = 0; ii < word.characters.size(); ++ii) {
    ss << word.characters[ii].value;
  }
  return ss.str().c_str();
}

void drawCharacters(std::vector<Character> characters) {
  for (size_t ii = 0; ii < characters.size(); ++ii) {
    Character character = characters[ii];
    cv::Rect rect = character.rect;
    char value = character.value;
    int group = character.group;
    int center_x = rect.width/2 + rect.x; int center_y = rect.height/2 + rect.y;
    std::stringstream ss; ss << value;
    cv::Size text_size = getTextSize(ss.str().c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, 1, NULL);
    putText(canvas, ss.str().c_str(), cv::Point(rect.x + rect.width/2 - text_size.width/2, rect.y + rect.height/2 + text_size.height/2), 
        cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, GREY, 1, CV_AA);
    cv::rectangle(canvas, rect, COLORS[group], 3);
  }
  cv::imshow(WINDOW_NAME, canvas);
}

Character& firstCharacter(Word& word) {
  int x = std::numeric_limits<int>::max();
  Character& first = word.characters[0];
  for (size_t ii = 0; ii < word.characters.size(); ++ii) {
    if (word.characters[ii].rect.x < x) {
      first = word.characters[ii];
      x = first.rect.x;
    }
  }
  return first;
}

Character& lastCharacter(Word& word) {
  int x = std::numeric_limits<int>::min();
  Character& last = word.characters[0];
  for (size_t ii = 0; ii < word.characters.size(); ++ii) {
    if (word.characters[ii].rect.x > x) {
      last = word.characters[ii];
      x = last.rect.x;
    }
  }
  return last;
}

double characterDistance(Character& a, Character& b) {
  // Scaled in y by VERTICAL_SENSITIVITY
  return pow(pow((a.rect.x+a.rect.width/2) - (b.rect.x+b.rect.width/2), 2) +
             pow((a.rect.y+a.rect.height/2) - (b.rect.y+b.rect.height/2), 2)*VERTICAL_SENSITIVITY,
         0.5);
}

double dissimilarity(Word a, Word b) {
  // Two words are similar (and should be merged) if:
  //  - They're on the same horizontal line
  //  - Their edges are close together:
  //      Word a's left char is close to Word b's right char or
  //      Word b's left char is close to Word a's right char
  double horizontal_distance = (MIN(characterDistance(firstCharacter(a), lastCharacter(b)),
                                characterDistance(firstCharacter(b), lastCharacter(a))));
  return horizontal_distance;
}

std::vector<Word> mergeWords(std::vector<Word> words, int word1idx, int word2idx) {
  std::vector<Word> merged_words;
  for (size_t ii = 0; ii < words.size(); ++ii) {
    if (ii != word1idx && ii != word2idx) {
      merged_words.push_back(words[ii]);
    }
  }
  Word merged_word;
  merged_word.characters.insert(merged_word.characters.end(), words[word1idx].characters.begin(), words[word1idx].characters.end());
  merged_word.characters.insert(merged_word.characters.end(), words[word2idx].characters.begin(), words[word2idx].characters.end());
  merged_words.push_back(merged_word);
  return merged_words;
}

std::vector<Word> clusterCharacters(std::vector<Character> characters) {
  std::vector<Word> words; // the final clusters
  // 1. Assign each character to its own cluster
  for (size_t ii = 0; ii < characters.size(); ++ii) {
    Word cur_word;
    cur_word.characters.push_back(characters[ii]);
    words.push_back(cur_word);
  }
  // 2. Combine clusters based on the best dissimilarity
  for (size_t ii = 0; ii < MAX_ITERATIONS; ++ii) {
    double best_dissimilarity = std::numeric_limits<double>::max();
    int word1idx = -1, word2idx = -1;
    for (int jj = 0; jj < words.size(); ++jj) {
      for (int kk = 0; kk < words.size(); ++kk) {
        if (jj == kk) { continue; }
        double cur_similarity = dissimilarity(words[jj], words[kk]);
        if (cur_similarity < best_dissimilarity) {
          best_dissimilarity = cur_similarity;
          word1idx = jj; word2idx = kk;
        }
      }
    }
    if (best_dissimilarity <= MAX_DISSIMILARITY) {
      std::cerr << "Combining " << getWordString(words[word1idx]);
      std::cerr << " and " << getWordString(words[word2idx]);
      std::cerr << " (dissimilarity = " << best_dissimilarity << ")" << std::endl;
      words = mergeWords(words, word1idx, word2idx);
    } else {
      std::cerr << "No more similar words -- breaking after " << ii << " iterations!" << std::endl;
      break;
    }
  }
  // 3. Assign the groups to the characters in the group
  for (size_t ii = 0; ii < words.size(); ++ii) {
    for (size_t jj = 0; jj < words[ii].characters.size(); ++jj) {
      Character& cur_character = words[ii].characters[jj];
      cur_character.group = ii + 1;
    }
  }
  return words;
}

void clearCanvas() {
  cv::rectangle(canvas, cv::Rect(0, 0, 1000, 800), cv::Scalar(255, 255, 255), -1);
}

void onMouse(int event, int x, int y, int, void* user) {
  if (event != cv::EVENT_LBUTTONDOWN) { return; }
  clearCanvas();
  std::vector<Character>* characters = (std::vector<Character>*) user;
  std::vector<Word> clustered_words = clusterCharacters(*characters);
  for (size_t ii = 0; ii < clustered_words.size(); ++ii) {
    drawCharacters(clustered_words[ii].characters);
  }
}

void onTrackbar(int event, void* user) { }

int main(int argc, char** argv) {
  std::vector<Character> chArr[] = {
    getCharacters("180", 20, 20),
    getCharacters("Pt", 160, 20),
    getCharacters("Beware", 20, 80),
    getCharacters("Test1", 20, 200),
    getCharacters("Test1", 200, 200),
    getCharacters("Test2", 20, 300),
    getCharacters("Test2", 210, 300),
    getCharacters("Test3", 20, 400),
    getCharacters("Test3", 220, 400),
    getCharacters("Test1", 500, 100),
    getCharacters("Test1", 500, 150),
    getCharacters("Test1", 500, 200),
    getCharacters("Test2", 750, 100),
    getCharacters("Test2", 750, 130),
    getCharacters("Test2", 750, 160),
    getCharacters("Test3", 500, 300),
    getCharacters("Test3", 500, 320),
    getCharacters("Test3", 500, 340),
    getCharacters("Test4", 750, 300),
    getCharacters("Test4", 750, 310),
    getCharacters("Test4", 750, 320),
  };

  std::vector<Character> chAll;
  for (size_t ii = 0; ii < sizeof(chArr)/sizeof(std::vector<Character>); ++ii) {
    chAll.insert(chAll.end(), chArr[ii].begin(), chArr[ii].end());
  }

  clearCanvas();
  drawCharacters(chAll);

  cv::setMouseCallback(WINDOW_NAME, onMouse, &chAll);
  cv::createTrackbar("Clustering Threshold", WINDOW_NAME, &MAX_DISSIMILARITY, 200, onTrackbar, &chAll); 
  cv::waitKey(0);

  return 0;
}
