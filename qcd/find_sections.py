import nltk

if __name__ == '__main__':
    import re

    with open('docs/Sweller2011_Chapter_TheElementInteractivityEffect.txt', 'r') as f:
        text = f.read()

    """Create a parser."""
    grammar = r"""
                NBAR:
                    {<DT>?<NN.*|JJ>*<NN.*>} # Nouns and Adjectives, terminated with Nouns

                NP:
                    {<NBAR>(<IN|CC><NBAR>)*}  # Above, connected with in/of/etc...
            """
    chunker = nltk.RegexpParser(grammar)

    pattern = re.compile(r"^(\d(\.\d)*\s+)?[A-Z]\w*(\s(\&|\w+))*$", flags=re.MULTILINE)

    sections = []

    for match in re.finditer(pattern, text):
        start, end = match.span()
        sections.append((match.group(0), end + 1))
        # print(match.group(0), match.span(), text[start:end])

    for i, title_index_pair in enumerate(sections):
        section, start = title_index_pair

        # tags = nltk.pos_tag(nltk.word_tokenize(section))
        # tree = chunker.parse(tags)
        #
        # # Filter out section titles that look like sentences not section titles...
        # if tree.subtrees(lambda st: not st.label().startswith('NN')):
        #     pass
        # else:
        #     print(section, 'POSSIBLY NOT SECTION TITLE')
        #     continue

        end = sections[i + 1][1] - 1 if i + 1 < len(sections) else len(text) - 1
        print(section, 'start:', start, 'end:', end)
