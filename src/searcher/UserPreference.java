/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package searcher;

/**
 *
 * @author anirban
 */
public class UserPreference {
    public int      queryNo;    // Query no. i.e. user no.
    public String[] docId;  // List of user preferences (use this 'docId' as document IDs such as "TRECCS-00086308-160" or tags such as "live-music")
    public int[]    rating; // List of user preferences (ratings of concerned documents)
    public int      nPreference;    // Size of the list
    public String[]    clusterId;      // clusterId of docs or tags based on 'queryTags_rated_3_4_Uniq_113_Phase2_clusters'
}
