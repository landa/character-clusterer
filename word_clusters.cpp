#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <string.h>
#include <limits.h>
#include <stdio.h>

// --- Clustering Parameters ---------------------------------------------------

int MAX_DISTANCE = 35.0;
const int MAX_ITERATIONS = 1000;
const double VERTICAL_SCALING = 1.5;

const int UNASSIGNED_GROUP = 0;

// --- Character and Word Structs ----------------------------------------------

typedef struct {
  cv::Rect rect;
  char value;
  int group;
} Character;

typedef struct {
  std::vector<Character> characters;
} Word;

// --- Visualization Variables and Functions -----------------------------------

const char* WINDOW_NAME = "Clustering Characters";
cv::Mat canvas(800, 1000, CV_8UC3);

cv::Scalar rc() { // get a random pastel color
  return cv::Scalar(150 + (float)rand()/((float)RAND_MAX/(100)),
                    150 + (float)rand()/((float)RAND_MAX/(100)),
                    150 + (float)rand()/((float)RAND_MAX/(100)));
}

const cv::Scalar GREY(100, 100, 100);
const cv::Scalar BLUE(255, 0, 0);
const cv::Scalar GREEN(0, 255, 0);
const cv::Scalar RED(0, 0, 255);
const cv::Scalar YELLOW(0, 220, 255);
const cv::Scalar CYAN(255, 255, 0);
const cv::Scalar MAGENTA(255, 0, 255);
const cv::Scalar BLACK(0, 0, 0);
const cv::Scalar COLORS[] = { GREY, rc(), rc(), rc(), rc(), rc(), rc(), rc(),
                              rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(),
                              rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(),
                              rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(),
                              rc(), rc(), rc(), rc(), rc(), rc(), rc(), rc(),
                              rc(), rc(), rc(), rc(), rc(), rc() };

const int CHARACTER_WIDTH = 30;
const int CHARACTER_HEIGHT = 40;
const int CHARACTER_SEPARATION = 35;

// --- Helper Methods (Testing and Visualization) ------------------------------

std::vector<Character> generateCharacters(char* str, int offset_x, int offset_y) {
  std::vector<Character> characters;
  for (size_t ii = 0; ii < strlen(str); ++ii) {
    Character cur;
    cur.rect = cv::Rect(offset_x + ii*CHARACTER_SEPARATION, offset_y, CHARACTER_WIDTH, CHARACTER_HEIGHT);
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
    putText(canvas, ss.str().c_str(),
            cv::Point(rect.x + rect.width/2 - text_size.width/2, rect.y + rect.height/2 + text_size.height/2), 
            cv::FONT_HERSHEY_COMPLEX_SMALL, 1.2, GREY, 1, CV_AA);
    cv::rectangle(canvas, rect, COLORS[group], 3);
  }
  cv::imshow(WINDOW_NAME, canvas);
}

void clearCanvas() {
  cv::rectangle(canvas, cv::Rect(0, 0, 1000, 800), cv::Scalar(255, 255, 255), -1);
}

// --- Clusterer ---------------------------------------------------------------

Character firstCharacter(Word const word) {
  int x = std::numeric_limits<int>::max();
  Character first = word.characters[0];
  for (size_t ii = 0; ii < word.characters.size(); ++ii) {
    if (word.characters[ii].rect.x < x) {
      first = word.characters[ii];
      x = first.rect.x;
    }
  }
  return first;
}

Character lastCharacter(Word const word) {
  int x = std::numeric_limits<int>::min();
  Character last = word.characters[0];
  for (size_t ii = 0; ii < word.characters.size(); ++ii) {
    if (word.characters[ii].rect.x > x) {
      last = word.characters[ii];
      x = last.rect.x;
    }
  }
  return last;
}

double bottomEdge(Word const word) {
  double y = std::numeric_limits<int>::min();
  for (size_t ii = 0; ii < word.characters.size(); ++ii) {
    if (word.characters[ii].rect.y + word.characters[ii].rect.height > y) {
      y = word.characters[ii].rect.y + word.characters[ii].rect.height;
    }
  }
  return y;
}

double topEdge(Word const word) {
  double y = std::numeric_limits<int>::max();
  for (size_t ii = 0; ii < word.characters.size(); ++ii) {
    if (word.characters[ii].rect.y + word.characters[ii].rect.height < y) {
      y = word.characters[ii].rect.y + word.characters[ii].rect.height;
    }
  }
  return y;
}

double characterHorizontalDistance(Character const a, Character const b) {
  return MIN(
          MIN(
            abs((a.rect.x) - (b.rect.x)),
            abs((a.rect.x+a.rect.width) - (b.rect.x))
          ),
          MIN(
            abs((a.rect.x+a.rect.width) - (b.rect.x+b.rect.width)),
            abs((a.rect.x) - (b.rect.x+b.rect.width))
          )
         );
}

double wordHorizontalDistance(Word const a, Word const b) {
  return MIN(
          MIN(characterHorizontalDistance(firstCharacter(a), lastCharacter(b)),
             characterHorizontalDistance(firstCharacter(b), lastCharacter(a))),
          MIN(characterHorizontalDistance(firstCharacter(a), firstCharacter(b)),
             characterHorizontalDistance(lastCharacter(a), lastCharacter(b)))
         );
}

double wordVerticalDistance(Word const a, Word const b) {
  return MIN(
          MIN(abs(topEdge(a) - bottomEdge(b)), abs(topEdge(b) - bottomEdge(a))),
          MIN(abs(topEdge(a) - topEdge(b)), abs(bottomEdge(b) - bottomEdge(a)))
         );
}

double distance(Word const a, Word const b) {
  // Two words are similar (and should be merged) if:
  //  - They're on the same horizontal line
  //  - Their edges are close together:
  //      Word a's left char is close to Word b's right char or
  //      Word b's left char is close to Word a's right char
  double horizontal_distance = wordHorizontalDistance(a, b);
  double vertical_distance = VERTICAL_SCALING*wordVerticalDistance(a, b);
  return pow(pow(horizontal_distance, 2) + pow(vertical_distance, 2), 0.5);
}

std::vector<Word> mergeWords(std::vector<Word> words, int word1idx, int word2idx) {
  std::vector<Word> merged_words;
  for (size_t ii = 0; ii < words.size(); ++ii) {
    if (ii != word1idx && ii != word2idx) {
      merged_words.push_back(words[ii]);
    }
  }
  Word merged_word;
  merged_word.characters.insert(merged_word.characters.end(),
                                words[word1idx].characters.begin(),
                                words[word1idx].characters.end());
  merged_word.characters.insert(merged_word.characters.end(),
                                words[word2idx].characters.begin(),
                                words[word2idx].characters.end());
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
  // 2. Combine clusters based on the best distance
  for (size_t ii = 0; ii < MAX_ITERATIONS; ++ii) {
    double best_distance = std::numeric_limits<double>::max();
    int word1idx = -1, word2idx = -1;
    for (int jj = 0; jj < words.size(); ++jj) {
      for (int kk = 0; kk < words.size(); ++kk) {
        if (jj == kk) { continue; }
        double cur_similarity = distance(words[jj], words[kk]);
        if (cur_similarity < best_distance) {
          best_distance = cur_similarity;
          word1idx = jj; word2idx = kk;
        }
      }
    }
    if (best_distance <= MAX_DISTANCE) {
      std::cerr << "Combining " << getWordString(words[word1idx]);
      std::cerr << " and " << getWordString(words[word2idx]);
      std::cerr << " (distance = " << best_distance << ")" << std::endl;
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

// --- Tests -------------------------------------------------------------------

void testDistances() {
  std::cerr << std::endl << "--- Test distances ------------------" << std::endl;
  Word wa, wb;
  std::vector<Character> ca = generateCharacters("ab", 800, 500);
  wa.characters = ca;
  std::vector<Character> cb = generateCharacters("cde", 800, 500);
  wb.characters = cb;

  std::cerr << "Horizontal distance: " << wordHorizontalDistance(wa, wb) << std::endl;
  std::cerr << "Vertical distance: " << wordVerticalDistance(wa, wb) << std::endl;
  std::cerr << "Total distance: " << distance(wa, wb) << std::endl;
}

void testVerticalStack() {
  std::cerr << std::endl << "--- Test vertical stack -------------" << std::endl;
  // Three "Vest3" words, one below the other
  // Combining the two top ones should give the same distance to the
  // bottom one as the distance between the bottom one and the one
  // above it.
  Word wa, wb, wc, wab;
  std::vector<Character> ca = generateCharacters("Vest3", 500, 300);
  wa.characters = ca;
  std::vector<Character> cb = generateCharacters("Vest3", 500, 320);
  wb.characters = cb;
  std::vector<Character> cc = generateCharacters("Vest3", 500, 340);
  wc.characters = cc;
  wab.characters.insert(wab.characters.end(), ca.begin(), ca.end());
  wab.characters.insert(wab.characters.end(), cb.begin(), cb.end());

  std::cerr << "a,b distance: " << distance(wa, wb) << std::endl;
  std::cerr << "b,c distance: " << distance(wb, wc) << std::endl;
  std::cerr << "a,c distance: " << distance(wa, wc) << std::endl;
  std::cerr << "ab,c distance: " << distance(wab, wc) << std::endl;
}

void testHorizontalWords() {
  std::cerr << std::endl << "--- Test horizontal words -----------" << std::endl;
  // The characters of the word "Ves"
  // The distances between the characters should be lower than the
  // vertical separation from testVerticalStack.
  Word wa, wb, wc, wabc;
  std::vector<Character> ca = generateCharacters("V", 500, 300);
  wa.characters = ca;
  std::vector<Character> cb = generateCharacters("e", 500+CHARACTER_SEPARATION, 300);
  wb.characters = cb;
  std::vector<Character> cc = generateCharacters("s", 500+CHARACTER_SEPARATION*2, 300);
  wc.characters = cc;
  wabc.characters.insert(wabc.characters.end(), ca.begin(), ca.end());
  wabc.characters.insert(wabc.characters.end(), cb.begin(), cb.end());
  wabc.characters.insert(wabc.characters.end(), cc.begin(), cc.end());

  std::cerr << "V,e distance: " << wordHorizontalDistance(wa, wb) << std::endl;
  std::cerr << "e,s distance: " << wordHorizontalDistance(wb, wc) << std::endl;
  std::cerr << "V,s distance: " << wordHorizontalDistance(wa, wc) << std::endl;
}

void runTests() {
  // testDistances();
  // testVerticalStack();
  testHorizontalWords();
}

// --- Main --------------------------------------------------------------------

void onMouse(int event, int x, int y, int, void* user) {
  if (event != cv::EVENT_LBUTTONDOWN) { return; }
  clearCanvas();
  std::vector<Character>* characters = (std::vector<Character>*) user;
  std::vector<Word> clustered_words = clusterCharacters(*characters);
  for (size_t ii = 0; ii < clustered_words.size(); ++ii) {
    drawCharacters(clustered_words[ii].characters);
  }
}

int main(int argc, char** argv) {
  std::vector<Character> chArr[] = {
    generateCharacters("180", 20, 20),
    generateCharacters("Pt", 160, 20),
    generateCharacters("Beware", 20, 80),
    generateCharacters("Hest1", 20, 200),
    generateCharacters("Hest1", 200, 200),
    generateCharacters("Hest2", 20, 300),
    generateCharacters("Hest2", 210, 300),
    generateCharacters("Hest3", 20, 400),
    generateCharacters("Hest3", 220, 400),
    generateCharacters("Hest4", 20, 500),
    generateCharacters("Hest4", 240, 500),
    generateCharacters("Vest1", 500, 100),
    generateCharacters("Vest1", 500, 150),
    generateCharacters("Vest1", 500, 200),
    generateCharacters("Vest2", 750, 100),
    generateCharacters("Vest2", 750, 130),
    generateCharacters("Vest2", 750, 160),
    generateCharacters("Vest3", 500, 300),
    generateCharacters("Vest3", 500, 320),
    generateCharacters("Vest3", 500, 340),
    generateCharacters("Vest4", 750, 300),
    generateCharacters("Vest4", 750, 310),
    generateCharacters("Vest4", 750, 320),
  };

  runTests();

  std::vector<Character> chAll;
  for (size_t ii = 0; ii < sizeof(chArr)/sizeof(std::vector<Character>); ++ii) {
    chAll.insert(chAll.end(), chArr[ii].begin(), chArr[ii].end());
  }

  clearCanvas();
  drawCharacters(chAll);

  cv::setMouseCallback(WINDOW_NAME, onMouse, &chAll);
  cv::createTrackbar("Clustering Threshold", WINDOW_NAME, &MAX_DISTANCE, 200, NULL, &chAll); 
  cv::waitKey(0);

  return 0;
}
