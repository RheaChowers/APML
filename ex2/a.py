
    # data, words, pos = load_data()


    # pos = ["N", "V", "A"]
    # words = ["cat", "dog", "nice", "mouse", "fish", "eat"]
    # train = [
    #     (["N", "V", "N"], ["cat", "eat", "mouse"]),
    # (["N", "V", "A", "N"], ["dog", "eat", "nice", "cat"]),
    #  (["A", "N", "V", "N"], ["nice", "mouse", "eat", "fish"]),
    # ( ["N", "V", "A","N"],  ["cat", "eat", "nice","fish"]),
    # ( ["N", "V", "A","N"],  ["cat", "eat", "nice","fish"]),
    # ( ["N", "V", "A","N"],  ["cat", "eat", "nice","fish"]),
    # ( ["N", "V", "A","N"],  ["cat", "eat", "nice","fish"]),
    # (["N", "V", "A", "N"], ["dog", "eat", "nice", "cat"]),
    # (["N", "V", "A", "N"], ["dog", "eat", "nice", "cat"]),
    # (["N", "V", "A", "N"], ["dog", "eat", "nice", "cat"]),
    # (["N", "V", "N"], ["cat", "eat", "mouse"]),
    # (["A", "N", "V", "N"], ["nice", "mouse", "eat", "cat"]),
    # (["A", "N", "V", "N"], ["nice", "dog", "eat", "mouse"]),
    # (["A", "N", "V", "N"], ["nice", "cat", "eat", "fish"]),
    # ]


    # _, _, data = edit_input(words, pos, data)
    # random.shuffle(data)
    # samps = math.ceil(len(data) * 0.1)
    # train = data[:samps]
    # test_samps = math.ceil(len(data) * 0.9)
    # test = data[test_samps:]

    # mod = HMM(pos, words, train)
    # sentences = [s[1] for s in test]
    # tagies = [s[0] for s in test]
    # sentences = [s[1] for s in train]
    # tagies = [s[0] for s in train]

    # mod = MEMM(pos, words, train)
    # tot = 0
    # cor = 0
    # print("started checking accuracy....\n")
    # for i, tag in enumerate(mod.viterbi(sentences, mod.w)):
    #     # print(mod.viterbi(sentences, mod.w))
    #     for j, t in enumerate(tag):
    #         tot += 1
    #         if tag[j] == tagies[i][j]:
    #             cor += 1
    # print("the accuracy is :", cor/tot)