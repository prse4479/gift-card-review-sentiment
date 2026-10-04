#!/usr/bin/env python3
"""
AMAZON GIFT CARD REVIEW SENTIMENT CLASSIFIER
(beginner-friendly version of classify_gift_cards.py)

WHAT THIS PROGRAM DOES
----------------------
Amazon sells gift cards, and customers leave reviews on them.
This program reads every review, looks at the review's TITLE and TEXT,
and decides whether the review is:
    - "positive"  (the customer liked it)
    - "negative"  (the customer had a problem)
    - "neutral"   (the text gives us no feelings to work with)

HOW IT MAKES ITS DECISION (3 steps, in order)
---------------------------------------------
Step 1 - WORD LIST CHECK
    We have three lists of phrases:
        - complaint phrases   -> if found, label is "negative"
        - rhetorical phrases  -> if found, label is "positive"
        - factual praise      -> if found, label is "positive"
    We check the complaint list FIRST, because a review can say
    "arrived on time but the card is torn" -- one good part, one bad part.
    The bad part should win.

Step 2 - SENTIMENT SCORING (VADER)
    If no phrase from the lists was found, we ask a small program called
    VADER to score the text. VADER has a dictionary of ~7,500 English words
    with feelings attached ("great" = positive, "awful" = negative).
    VADER returns one number, the "compound score", between -1 and 1:
        - between 0 and 1  -> positive feelings  -> label "positive"
        - between -1 and 0 -> negative feelings  -> label "negative"

Step 3 - NO FEELING AT ALL
    If the score is exactly 0, the text had no feeling words at all
    (example: "As advertised"). We label it "neutral" instead of guessing.

IMPORTANT: the star rating is NEVER used. Only the words you type matter.

HOW TO RUN THIS PROGRAM
-----------------------
    python3 classify_gift_cards_explained.py

    (To use different files, add them after the command:
     python3 classify_gift_cards_explained.py my_reviews.jsonl.gz my_output.jsonl.gz)
"""

# ----------------------------------------------------------------------------
# PART 1 - IMPORT THE TOOLS WE NEED
# ----------------------------------------------------------------------------

import json  # json.loads: turn one line of JSON text into a dictionary.
             # json.dumps: turn a dictionary back into one line of text.
import gzip  # lets us open ".gz" (compressed) files directly.
import sys   # sys.argv: the extra words typed after "python3 my_file.py".

# VADER is the sentiment-scoring helper. It is NOT part of Python itself,
# so we install it separately (pip3 install vaderSentiment).
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ----------------------------------------------------------------------------
# PART 2 - CHOOSE THE INPUT AND OUTPUT FILES
# ----------------------------------------------------------------------------

# The program will use these files, unless you typed different names
# on the command line (see the "HOW TO RUN" section above).
INPUT_FILE = "/Users/preethiselvaganapathy/Downloads/AI & LLM/Gift_Cards.jsonl.gz"
OUTPUT_FILE = "/Users/preethiselvaganapathy/Downloads/AI & LLM/Gift_Cards_classified_explained.jsonl.gz"

# ----------------------------------------------------------------------------
# PART 3 - THE THREE LISTS OF PHRASES
# ----------------------------------------------------------------------------
#
# A "phrase" here is just a piece of text (a word or a few words).
# For every review, we check: "does this review contain this phrase?"
# We search inside the review text written in lowercase, so
# "Does Not Work" and "does not work" both match.

# 1) COMPLAINT PHRASES -> the customer is unhappy.
#    We added many different ways people say the same thing,
#    including common misspellings ("satisffied") on purpose.
COMPLAINT_PHRASES = [
    # the card does not work
    "does not work", "doesn't work", "dont work", "do not work", "don't work",
    "did not work", "didn't work", "not working", "stopped working", "never worked",
    "won't work", "wont work", "wouldn't work", "would not work",
    # the card was not activated
    "does not activate", "not activated", "not active", "never activated",
    "won't activate", "wont activate", "didn't activate", "never activate",
    # the card never arrived
    "never received", "did not receive", "didn't receive", "never got", "never got it",
    "never arrived", "did not arrive", "didn't arrive", "never came", "never showed",
    "never showed up", "only received", "not received",
    # delivery problems (these must beat the positive "on time" phrase)
    "not on time", "wasn't on time", "was not on time", "arrived late", "came late",
    "delivered late",
    "wrong address", "wrong item", "wrong card", "wrong order", "wrong house",
    # arrived, but not as promised
    "not as expected", "wasn't as expected", "was not as expected",
    "not what i expected", "wasn't what i expected", "not what was expected",
    "not as described", "wasn't as described", "not as advertised",
    "not as shown", "not as pictured", "not as listed",
    # physical damage
    "ripped", "torn", "shredded", "damaged", "broken", "bent", "crushed",
    "too small", "not valid", "invalid", "expired",
    # wrong amount or a scam
    "shorted", "missing", "scam", "fraud", "fake", "counterfeit", "compromised",
    "deactivate", "deactivated", "desactivate",
    # the customer regrets the purchase or is confused by it
    "don't buy", "do not buy", "dont buy",
    "can't figure out", "cant figure out", "cannot figure out",
    # feeling words that VADER's dictionary does not know
    "useless", "junk", "waste of money", "hassle", "nightmare", "refund",
    "not satisfied", "not satisffied", "unsatisfied", "not pleased", "not happy",
    "dissapoint",   # catches the misspelling of "disappointed"
    "frustrat",     # catches "frustrated" and "frustrating"
]

# 2) RHETORIC PHRASES -> positive, but said in a tricky way.
#    "What's not to like?" actually means "everything is likeable",
#    but the word "not" makes simple word-scoring programs confused.
RHETORIC_POSITIVE_PHRASES = [
    "what's not to like", "what is not to like", "whats not to like",
    "what's not to love", "what is not to love", "whats not to love",
    "can't go wrong", "cant go wrong", "cannot go wrong", "can't go too wrong",
    "can't beat", "cant beat", "can't be beat", "cant be beat", "hard to beat",
    "no complaints", "can't complain", "cant complain", "nothing to complain",
    "no issues", "no problems", "no problem", "no trouble", "without issue",
    "who doesn't love", "who doesnt love", "who wouldn't love", "who wouldnt love",
    "who doesn't like", "who doesnt like", "who wouldn't want", "who wouldnt want",
    "who doesn't want", "who doesnt want",
]

# 3) FACTUAL PRAISE PHRASES -> positive, but stated using only plain facts.
#    "Delivered on time and as described" has NO feeling words at all,
#    yet the customer clearly means it positively.
FACTUAL_POSITIVE_PHRASES = [
    "as described", "as advertised", "just as described", "exactly as described",
    "exactly as advertised", "as pictured", "as shown", "as listed",
    "as expected", "just as expected", "exactly as expected",
    "on time", "arrived", "delivered", "received",
    "quick delivery", "fast delivery", "fast shipping", "quick shipping",
    "shipped fast", "shipped quickly", "came quickly", "came fast", "arrived fast",
    "works", "worked", "it works", "they work", "work fine", "works fine",
    "worked fine", "works as expected", "worked as expected",
    "works perfectly", "worked perfectly", "everything works", "everything worked",
    "just what i needed", "just what i wanted", "exactly what i needed",
    "exactly what i wanted",
]

# ----------------------------------------------------------------------------
# PART 4 - HELPER FUNCTIONS (small jobs, written one at a time)
# ----------------------------------------------------------------------------


def find_phrase_label(lowercase_text):
    """
    Search the review text for any phrase from our lists.

    Returns:
        "negative"  if a complaint phrase is found,
        "positive"  if a rhetoric or factual-praise phrase is found,
        None        if no phrase from any list is found.
    """
    # The `in` operator asks: "does the phrase appear inside the text?"
    # Example: "not activated" in "card was not activated" -> True
    for phrase in COMPLAINT_PHRASES:
        if phrase in lowercase_text:
            return "negative"

    # Only if no complaint was found do we look for positive phrases.
    for phrase in RHETORIC_POSITIVE_PHRASES:
        if phrase in lowercase_text:
            return "positive"

    for phrase in FACTUAL_POSITIVE_PHRASES:
        if phrase in lowercase_text:
            return "positive"

    # Nothing matched. The value None means "no answer from the lists yet".
    return None


def classify_review(analyzer, combined_text):
    """
    Decide the label for one review, using the 3-step plan:
    phrase lists first, then VADER, then neutral.

    Arguments:
        analyzer      - the VADER scoring tool (created once, in main)
        combined_text - the review's title + text, in one string

    Returns:
        (label, score) - for example ("positive", 0.8912)
    """
    # Step 0: if the review has no title and no text, there is nothing
    # to judge. Say neutral instead of guessing.
    if combined_text == "":
        return "neutral", 0.0

    # Searching is case-sensitive, so first make a lowercase copy of the
    # text. This way "Great" and "great" are treated the same way.
    lowercase_text = combined_text.lower()

    # Step 1: try the phrase lists.
    phrase_label = find_phrase_label(lowercase_text)
    if phrase_label is not None:
        # A phrase matched. Ask VADER for a score anyway, so every
        # review in the output file has a number next to its label.
        score = analyzer.polarity_scores(combined_text)["compound"]
        return phrase_label, score

    # Step 2: no phrase matched, so use VADER's feeling score.
    # VADER returns a dictionary; we take the "compound" value from it,
    # which is one number between -1 (very negative) and 1 (very positive).
    score = analyzer.polarity_scores(combined_text)["compound"]

    if score > 0:
        return "positive", score   # the text leans positive
    if score < 0:
        return "negative", score   # the text leans negative

    # Step 3: score is exactly 0 -> VADER found no feeling words at all.
    return "neutral", 0.0


def normalize_quotes(text):
    """
    Replace curly quotes/apostrophes with straight ones.

    Many reviews are typed on phones, which change ' into a curly
    character like '. Our phrase lists use the straight ', so without
    this step "Didn't work" in a review would never match the phrase
    "didn't work", and the review would be mislabeled.

    We only change the COPY used for matching. The original review text
    is written to the output file unchanged.
    """
    # \u2019 and \u2018 are the curly right/left apostrophe,
    # \u201c and \u201d are curly double quotes.
    text = text.replace("\u2019", "'")
    text = text.replace("\u2018", "'")
    text = text.replace("\u201c", '"')
    text = text.replace("\u201d", '"')
    return text


def get_review_text(review):
    """
    Join the review's "title" and "text" into one string.
    If a key is missing (some reviews have no text), use an empty string
    so the program does not crash.
    """
    if "title" in review and review["title"] is not None:
        title = review["title"]
    else:
        title = ""

    if "text" in review and review["text"] is not None:
        text = review["text"]
    else:
        text = ""

    # Put the two pieces together with a space between them,
    # then remove any extra spaces at the very start or end.
    combined = title + " " + text
    combined = combined.strip()
    combined = normalize_quotes(combined)
    return combined

# ----------------------------------------------------------------------------
# PART 5 - THE MAIN FUNCTION (where the whole job happens, top to bottom)
# ----------------------------------------------------------------------------


def main():
    """
    Run the whole classification job:
    open file, read every review, classify it, write the results out.
    """

    # -- 5.1 Read the file names from the command line (optional) ------------
    # If the user typed extra file names after the program name, use them.
    # Example: python3 classify_gift_cards_explained.py data.jsonl.gz result.jsonl.gz
    # sys.argv is a list: ["classify_gift_cards_explained.py", "data.jsonl.gz", "result.jsonl.gz"]
    input_file_name = INPUT_FILE
    output_file_name = OUTPUT_FILE
    if len(sys.argv) >= 2:
        input_file_name = sys.argv[1]   # the second word on the command line
    if len(sys.argv) >= 3:
        output_file_name = sys.argv[2]  # the third word on the command line

    # -- 5.2 Set up the counting variables -----------------------------------
    # python can't do "positive: 0 + 1" inline, so we keep three counters
    # in a dictionary: label_counts["positive"] means "how many positives so far".
    label_counts = {"positive": 0, "negative": 0, "neutral": 0}
    phrase_hit_counts = {"positive": 0, "negative": 0}
    total_reviews = 0

    # Create the VADER analyzer once, before the loop. It is a big
    # dictionary of words, so we do not want to rebuild it for each review.
    analyzer = SentimentIntensityAnalyzer()

    # -- 5.3 Open the input file ----------------------------------------------
    # .gz files are compressed, so they need gzip.open; plain files use open.
    # The "with ... as ..." pattern is a safe way to open a file:
    # Python automatically closes it when the block of code finishes.
    if input_file_name.endswith(".gz"):
        input_file = gzip.open(input_file_name, "rt", encoding="utf-8")
    else:
        input_file = open(input_file_name, "r", encoding="utf-8")

    input_lines = []   # we will store every finished review line here,
                       # then write them all to the output file at the end

    with input_file as data_file:
        # The task repeats for EVERY line in the file. One line = one review.
        # This for-loop runs 152,410 times for the gift card dataset.
        for line in data_file:
            total_reviews += 1

            # Print a progress update every 50,000 reviews so we can
            # see the loop is alive while it works.
            if total_reviews % 50000 == 0:
                print("... processed", total_reviews, "reviews so far")

            # json.loads: turn the line's text into a dictionary.
            # Example: '{"title": "Great gift"}' becomes {"title": "Great gift"}
            review = json.loads(line)

            # Get the review's words in one lowercase-friendly string.
            combined_text = get_review_text(review)

            # Decide the label and score using our 3-step plan.
            label, score = classify_review(analyzer, combined_text)

            # Add the two new pieces of information to the review dictionary.
            review["sentiment"] = label          # "positive", "negative" or "neutral"
            review["sentiment_score"] = round(score, 4)   # keep 4 decimal places

            # Turn the dictionary back into one line of text, ready for the
            # output file. It looks exactly like the input line, plus the
            # two new "sentiment" fields.
            output_line = json.dumps(review)
            input_lines.append(output_line + "\n")

            # Update the counters.
            label_counts[label] = label_counts[label] + 1
            phrase_label = find_phrase_label(combined_text.lower())
            if phrase_label is not None:
                phrase_hit_counts[phrase_label] = phrase_hit_counts[phrase_label] + 1

    # -- 5.4 Write all the classified reviews to the output file ---------------
    with gzip.open(output_file_name, "wt", encoding="utf-8") as output_file:
        for output_line in input_lines:
            output_file.write(output_line)

    # -- 5.5 Print a short report about what happened --------------------------
    print()
    print("Total reviews:", total_reviews)
    print()
    print("Label counts:")
    for label in ["positive", "negative", "neutral"]:
        count = label_counts[label]
        # In Python 3, "/" always gives a decimal answer (0.33, not 0).
        # We multiply by 100 to turn a fraction into a percentage.
        percent = count * 100 / total_reviews
        print("  ", label, ":", count, "(", round(percent, 2), "%)")
    print()
    print("Phrase lists caught:", phrase_hit_counts)
    print("Output written to:", output_file_name)


# ----------------------------------------------------------------------------
# PART 6 - TELL PYTHON TO START
# ----------------------------------------------------------------------------

# Every Python file has a hidden variable called __name__.
#   - When you RUN this file (python3 classify_gift_cards_explained.py),
#     __name__ equals the string "__main__", so main() runs.
#   - When another program IMPORTS this file, __name__ equals the file
#     name instead, so main() does NOT run automatically.
# This one pattern lets a file be both a program and a reusable toolbox.
if __name__ == "__main__":
    main()
