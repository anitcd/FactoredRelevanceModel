# sh ani_grid_BM25_general.sh ani_param_BM25_k1_b_subset
# sh ani_grid_BM25_general.sh ani_param_BM25_k1_b

#topicfile="topics.401-450"
topicfile="topics.301-350"
#topicpath="/store/Data/TRECAdhoc/topics/topics.401-450"
topicpath="/store/Data/TRECAdhoc/topics/topics.301-350"
indexpath="/store/Data/TRECAdhoc/index"
#qrelpath="/store/Data/TRECAdhoc/qrels/qrels.trec8.adhoc.parts1-5"
qrelpath="/store/Data/TRECAdhoc/qrels/qrels.trec6.adhoc.parts1-5"

rm -f /store/Data/TRECAdhoc/BM25statsGeneral.txt

while read k1 b;
do

  #resfile="/home/anirban/"$topicfile"-full-content-BM25k1="$k1",b="$b
  resfile="/home/anirban/"$topicfile"-content-BM25k1="$k1",b="$b

  #./searcher_TREC8.sh $indexpath full-content $topicpath 1 1 $k1 $b
  ./searcher_TREC8.sh $indexpath content $topicpath 1 1 $k1 $b

  #ndcg5=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^ndcg_cut_5 " | awk '{print $3}')
  #ndcg=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^ndcg " | awk '{print $3}')
  P5=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^P_5 " | awk '{print $3}')
  P10=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^P_10 " | awk '{print $3}')
  map=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^map " | awk '{print $3}')

  echo "BM25_k1_"$k1"_b_"$b"\t"$P5"\t"$P10"\t"$map >> /store/Data/TRECAdhoc/BM25statsGeneral.txt
  echo "BM25: k1 = "$k1"\tb = "$b"\tdone!"

  #mv $resfile /store/Data/TRECAdhoc/resfiles/
  rm -f $resfile

done < $1

