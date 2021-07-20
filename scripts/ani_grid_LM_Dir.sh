# sh ani_grid_LM_Dir.sh ani_param_LMDi_mu

LM="Dirichlet"
#LM="Jelinek-Mercer"

rm -f /store/Data/TREC_CS/LMstats.txt

#topicfile="topic_61_Phase1_TRECformat"
topicfile="topic_61_Phase1_TRECformat"
indexpath="/store/Data/TREC_CS/poiAsTRECDoc2_Index"

while read p;
do

  resfile="/home/anirban/"$topicfile".xml-full-content-LM-"$LM$p".000000"
  #resfile="/home/anirban/"$topicfile".xml-full-content-LM-"$LM$p

  #./searcher.sh $indexpath full-content /store/Data/TREC_CS/$topicfile.xml 1 3 $p
  ./searcher_TRECCS_Novel.sh $indexpath full-content /store/Data/TREC_CS/$topicfile.xml 1 3 $p

  ndcg5=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt $resfile | grep "^ndcg_cut_5 " | awk '{print $3}')
  ndcg=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt $resfile | grep "^ndcg " | awk '{print $3}')
  P5=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt $resfile | grep "^P_5 " | awk '{print $3}')
  map=$(/store/Softwares/trec_eval.9.0/trec_eval -m 'all_trec' /store/Data/TREC_CS/qrels_TREC2016_CS.qrel.txt $resfile | grep "^map " | awk '{print $3}')

  echo "LM"$LM"_lamda_"$p"\t"$ndcg5"\t"$ndcg"\t"$P5"\t"$map >> /store/Data/TREC_CS/LMstats.txt
  echo "LM"$LM" lamda = "$p"\tdone!"

  mv $resfile /store/Data/TREC_CS/resfiles/

done < $1

