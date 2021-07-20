# sh ani_grid_LM_JM_general.sh ani_param_LMJM_lamda

LM="Jelinek-Mercer"

#topicfile="topics.401-450"
topicfile="topics.301-350"
#topicpath="/store/Data/TRECAdhoc/topics/topics.401-450"
topicpath="/store/Data/TRECAdhoc/topics/topics.301-350"
indexpath="/store/Data/TRECAdhoc/index"
#qrelpath="/store/Data/TRECAdhoc/qrels/qrels.trec8.adhoc.parts1-5"
qrelpath="/store/Data/TRECAdhoc/qrels/qrels.trec6.adhoc.parts1-5"

rm -f /store/Data/TRECAdhoc/LMJMstatsGeneral.txt

while read p;
do

  #resfile="/home/anirban/"$topicfile"-full-content-LM-"$LM$p
  resfile="/home/anirban/"$topicfile"-full-content-LM-"$LM$p

  #./searcher_TRECCS_Novel.sh $indexpath full-content /store/Data/TREC_CS/$topicfile.xml 1 2 $p
  #./searcher_TREC8.sh $indexpath full-content $topicpath 1 2 $p
  ./searcher_TREC8.sh $indexpath full-content $topicpath 1 2 $p

  #ndcg5=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^ndcg_cut_5 " | awk '{print $3}')
  #ndcg=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^ndcg " | awk '{print $3}')
  P5=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^P_5 " | awk '{print $3}')
  P10=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^P_10 " | awk '{print $3}')
  map=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^map " | awk '{print $3}')

  echo "LM"$LM"_lamda_"$p"\t"$P5"\t"$P10"\t"$map >> /store/Data/TRECAdhoc/LMJMstatsGeneral.txt
  echo "LM"$LM" lamda = "$p"\tdone!"

  #mv $resfile /store/Data/TREC_CS/resfiles/
  rm -f $resfile

done < $1

