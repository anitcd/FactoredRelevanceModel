# For indexing
# ./index-spec.sh /store/Data/TREC_CS/poiAsTRECDoc_init.properties /store/Data/TREC_CS/poiAsTRECDoc_Index


cd ..
ant
cd scripts/

# For retrieval and evaluation

# LM DIrichlet
#./searcher.sh /store/Data/TREC_CS/poiAsTRECDoc1_Index full-content /store/Data/TREC_CS/topic_61_Phase1_TRECformat.xml 1 3 1000
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/topic_61_Phase1_TRECformat.xml-full-content-LM-Dirichlet1000.000000

# LM Jelineck-Mercer
#./searcher.sh /store/Data/TREC_CS/poiAsTRECDoc1_Index full-content /store/Data/TREC_CS/topic_61_Phase1_TRECformat.xml 1 2 0.6
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/topic_61_Phase1_TRECformat.xml-full-content-LM-Jelinek-Mercer0.600000

# BM25
#./searcher.sh /store/Data/TREC_CS/poiAsTRECDoc_phase2_5K_Index full-content /store/Data/TREC_CS/topic_61_Phase1_TRECformat.xml 1 1 1.25 0.3
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/topic_61_Phase1_TRECformat.xml-full-content-BM25k1=1.25,b=0.3

#./searcher.sh /store/Data/TREC_CS/poiAsTRECDoc_phase2_5K_Index full-content /store/Data/TREC_CS/topic_58_Phase2_TRECformat.xml 1 1 1.25 0.3
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/topic_58_Phase2_TRECformat.xml-full-content-BM25k1=1.25,b=0.3

#./searcher.sh /store/Data/TREC_CS/poiAsTRECDoc_phase2_5K_Index full-content /store/Data/TREC_CS/topic_61_Phase1_TRECformat.xml 1 1 1.25 0.3
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/topic_61_Phase1_TRECformat.xml-full-content-BM25k1=1.25,b=0.3


#...... Testing W2V and multi ......
#./searcher.sh /store/Data/TREC_CS/poiAsTRECDoc2_Index full-content /store/Data/TREC_CS/topic_61_Phase1_TRECformat.xml 1 1 1.25 0.3
#./searcher_TRECCS.sh /store/Data/TREC_CS/poiAsTRECDoc2_Index full-content /store/Data/TREC_CS/topic_61_Phase1_TRECformat.xml 1 1 1.25 0.3
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/topic_61_Phase1_TRECformat.xml-full-content-BM25k1=1.25,b=0.3

#./searcher_TRECCS.sh /store/Data/TREC_CS/poiAsTRECDoc2_Index full-content /store/Data/TREC_CS/topic_58_Phase2_TRECformat.xml 1 1 1.25 0.3
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/topic_58_Phase2_TRECformat.xml-full-content-BM25k1=1.25,b=0.3
#...................................


#.......... Testing Debu ...........
# BM25
#./searcher_TRECCS_Novel.sh /store/Data/TREC_CS/poiAsTRECDoc2_Index full-content /store/Data/TREC_CS/topic_61_Phase1_TRECformat.xml 1 1 1.1 0.3
#./searcher_TRECCS_Novel.sh /store/Data/TREC_CS/poiAsTRECDoc_200K_Index full-content /store/Data/TREC_CS/topic_61_Phase1_TRECformat.xml 1 1 1.25 0.3
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/topic_61_Phase1_TRECformat.xml-full-content-BM25k1=1.1,b=0.3


# Geocoded
./searcher_TRECCS_Novel.sh /store/Data/TREC_CS/poiAsTRECDoc2_Index full-content /store/Data/TREC_CS/topic_61_Phase1_extendedX10_TRECformat.xml 1 1 1.1 0.3
/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel_extended_[0,4]_withoutGeo.txt /home/anirban/topic_61_Phase1_extendedX10_TRECformat.xml-full-content-BM25k1=1.1,b=0.3


#./searcher_TRECCS_Novel.sh /store/Data/TREC_CS/poiAsTRECDoc2_Index full-content /store/Data/TREC_CS/NeuMF_query_top10 1 1 1.1 0.3
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/NeuMF_query_top10-full-content-BM25k1=1.1,b=0.3

# LM-Dir
#./searcher_TRECCS_Novel.sh /store/Data/TREC_CS/poiAsTRECDoc2_Index full-content /store/Data/TREC_CS/topic_61_Phase1_TRECformat.xml 1 3 850
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/topic_61_Phase1_TRECformat.xml-full-content-LM-Dirichlet850.000000


# RM3
#./rblm.sh /store/Data/TREC_CS/poiAsTRECDoc2_Index /store/Data/TREC_CS/topic_61_Phase1_TRECformat.xml /home/anirban/12 1 25 0.4 1 1 1
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/12topic_61_Phase1_TRECformat.xml-BM25k1=1.1,b=0.3-D1-T20-rm3-0.4-full-content-full-content.res

#...................................

#./searcher.sh /store/Data/TREC_CS/poiAsTRECDoc1_Index full-content /store/Data/TREC_CS/topic_58_Phase2_TRECformat.xml 1 1 1.25 0.3
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/topic_58_Phase2_TRECformat.xml-full-content-BM25k1=1.25,b=0.3


#./searcher.sh /store/Data/TREC_CS/poiAsTRECDoc1_Index full-content /store/Data/TREC_CS/topic_61_Phase1_extendedX10_TRECformat.xml 1 3 1000
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/topic_61_Phase1_extendedX10_TRECformat.xml-full-content-LM-Dirichlet1000.000000



#./rblm.sh /store/Data/TREC_CS/poiAsTRECDoc1_Index /store/Data/TREC_CS/topic_61_Phase1_TRECformat.xml /home/anirban/12 1 10 0.4 1 1 3
#/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt /home/anirban/12topic_61_Phase1_TRECformat.xml-LMDirichlet2000.0-D1-T10-rm3-0.4-full-content-full-content.res


#./rblm.sh /store/Data/TREC_CS/poiAsTRECDoc_Index /store/Data/TREC_CS/topic_61_Phase1_extendedX10_TRECformat.xml /home/anirban/12 1 10 0.4 1 1 3



