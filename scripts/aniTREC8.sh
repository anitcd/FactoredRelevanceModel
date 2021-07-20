# For indexing
# ./index-spec.sh /store/Data/TREC_CS/poiAsTRECDoc_init.properties /store/Data/TREC_CS/poiAsTRECDoc_Index


cd ..
ant
cd scripts/


qrelTRECRobust="/store/Data/TRECAdhoc/qrels/qrels.robust2004.txt"

# For retrieval and evaluation TREC 8

# BM25	best MAP (0.5, 0.5)
#./searcher_TREC8.sh /store/Data/TRECAdhoc/index full-content /store/Data/TRECAdhoc/topics/topics.401-450 1 1 0.5 0.5

#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TRECAdhoc/qrels/qrels.trec8.adhoc.parts1-5 /home/anirban/topics.401-450-full-content-BM25k1=0.5,b=0.5

# BM25	best P@5 (0.9, 0.5)
#./searcher_TREC8.sh /store/Data/TRECAdhoc/index full-content /store/Data/TRECAdhoc/topics/topics.401-450 1 1 0.9 0.5

#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TRECAdhoc/qrels/qrels.trec8.adhoc.parts1-5 /home/anirban/topics.401-450-full-content-BM25k1=0.9,b=0.5


# LM Dir best MAP (400)
# LM Dir best MAP (mu=200) tuned for TREC 6
./searcher_TREC8.sh /store/Data/TRECAdhoc/index full-content /store/Data/TRECAdhoc/topics/topics.401-450 1 3 200

/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TRECAdhoc/qrels/qrels.trec8.adhoc.parts1-5 /home/anirban/topics.401-450-full-content-LM-Dirichlet200.000000


# LM JM
#./searcher_TREC8.sh /store/Data/TRECAdhoc/index full-content /store/Data/TRECAdhoc/topics/topics.401-450 1 2 0.400000

#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TRECAdhoc/qrels/qrels.trec8.adhoc.parts1-5 /home/anirban/topics.401-450-full-content-LM-Jelinek-Mercer0.400000


#--------------------------------------------------------------------------------------------------------------------



# TREC 6
# BM25	best MAP (0.5, 0.5) tuned for TREC 8
#./searcher_TREC8.sh /store/Data/TRECAdhoc/index content /store/Data/TRECAdhoc/topics/topics.301-350 1 1 0.5 0.5

#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TRECAdhoc/qrels/qrels.trec6.adhoc.parts1-5 /home/anirban/topics.301-350-content-BM25k1=0.5,b=0.5

# LM Dir best MAP (mu=200) tuned for TREC 6
#./searcher_TREC8.sh /store/Data/TRECAdhoc/index full-content /store/Data/TRECAdhoc/topics/topics.301-350 1 3 200

#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TRECAdhoc/qrels/qrels.trec6.adhoc.parts1-5 /home/anirban/topics.301-350-full-content-LM-Dirichlet200.000000

#--------------------------------------------------------------------------------------------------------------------

# TREC 7
# BM25	best MAP (0.5, 0.5) tuned for TREC 8
#./searcher_TREC8.sh /store/Data/TRECAdhoc/index full-content /store/Data/TRECAdhoc/topics/topics.351-400 1 1 0.5 0.5

#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TRECAdhoc/qrels/qrels.trec7.adhoc.parts1-5 /home/anirban/topics.351-400-full-content-BM25k1=0.5,b=0.5

# LM Dir best MAP (mu=200) tuned for TREC 6
#./searcher_TREC8.sh /store/Data/TRECAdhoc/index full-content /store/Data/TRECAdhoc/topics/topics.351-400 1 3 200

#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TRECAdhoc/qrels/qrels.trec7.adhoc.parts1-5 /home/anirban/topics.351-400-full-content-LM-Dirichlet200.000000


#--------------------------------------------------------------------------------------------------------------------

# TREC Robust 99 queries
# BM25	best MAP (0.5, 0.5) tuned for TREC 8
#./searcher_TREC8.sh /store/Data/TRECAdhoc/index full-content /store/Data/TRECAdhoc/topics/topics.601-700_99queries 1 1 0.5 0.5
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelTRECRobust /home/anirban/topics.601-700_99queries-full-content-BM25k1=0.5,b=0.5

# LM Dir best MAP (mu=200) tuned for TREC 6
#./searcher_TREC8.sh /store/Data/TRECAdhoc/index full-content /store/Data/TRECAdhoc/topics/topics.601-700_99queries 1 3 200

#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelTRECRobust /home/anirban/topics.601-700_99queries-full-content-LM-Dirichlet200.000000


