# sh ani_grid_LM_Dir_general.sh ani_param_LMDi_mu
# sh ani_grid_LM_Dir_general.sh ani_param_LMDi_mu_subset

LM="Dirichlet"
#LM="Jelinek-Mercer"

rm -f /store/Data/TRECAdhoc/LMDirstatsGeneral.txt

#topicfile="topics.401-450"
topicfile="topics.301-350"
#topicpath="/store/Data/TRECAdhoc/topics/topics.401-450"
topicpath="/store/Data/TRECAdhoc/topics/topics.301-350"
indexpath="/store/Data/TRECAdhoc/index"
#qrelpath="/store/Data/TRECAdhoc/qrels/qrels.trec8.adhoc.parts1-5"
qrelpath="/store/Data/TRECAdhoc/qrels/qrels.trec6.adhoc.parts1-5"

while read p;
do

  #resfile="/home/anirban/"$topicfile".xml-full-content-LM-"$LM$p
  #resfile="/home/anirban/"$topicfile"-full-content-LM-"$LM$p".000000"
  resfile="/home/anirban/"$topicfile"-content-LM-"$LM$p".000000"


  #./searcher.sh $indexpath full-content /store/Data/TREC_CS/$topicfile.xml 1 3 $p
  #./searcher_TRECCS_Novel.sh $indexpath full-content /store/Data/TREC_CS/$topicfile.xml 1 3 $p
  #./searcher_TREC8.sh $indexpath full-content $topicpath 1 3 $p
  ./searcher_TREC8.sh $indexpath content $topicpath 1 3 $p

  #ndcg5=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^ndcg_cut_5 " | awk '{print $3}')
  #ndcg=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^ndcg " | awk '{print $3}')
  P5=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^P_5 " | awk '{print $3}')
  P10=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^P_10 " | awk '{print $3}')
  map=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' $qrelpath $resfile | grep "^map " | awk '{print $3}')

  echo "LM"$LM"_Mu_"$p"\t"$P5"\t"$P10"\t"$map >> /store/Data/TRECAdhoc/LMDirstatsGeneral.txt
  echo "LM"$LM" Mu = "$p"\tdone!"

  #mv $resfile /store/Data/TREC_CS/resfiles/
  rm -f $resfile

done < $1

