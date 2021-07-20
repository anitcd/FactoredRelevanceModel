# sh ani_grid_BM25.sh ani_param_BM25_k1_b

topicfile="topic_61_Phase1_TRECformat"
#topicfile="topic_58_Phase2_TRECformat"
indexpath="/store/Data/TREC_CS/poiAsTRECDoc2_Index"

#rm -f /store/Data/TREC_CS/BM25stats.txt

while read k1 b;
do

  resfile="/home/anirban/"$topicfile".xml-full-content-BM25k1="$k1",b="$b

  ./searcher_TRECCS_Novel.sh $indexpath full-content /store/Data/TREC_CS/$topicfile.xml 1 1 $k1 $b

  ndcg5=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt $resfile | grep "^ndcg_cut_5 " | awk '{print $3}')
  ndcg=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt $resfile | grep "^ndcg " | awk '{print $3}')
  P5=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt $resfile | grep "^P_5 " | awk '{print $3}')
  map=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt $resfile | grep "^map " | awk '{print $3}')

  echo "BM25_k1_"$k1"_b_"$b"\t"$ndcg5"\t"$ndcg"\t"$P5"\t"$map >> /store/Data/TREC_CS/BM25stats.txt
  echo "BM25: k1 = "$k1"b = "$b"\tdone!"

  mv $resfile /store/Data/TREC_CS/resfiles/

done < $1

