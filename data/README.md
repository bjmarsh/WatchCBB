### Description of contents
1. `teams.csv` - information on all D-I college basketball teams since 2000. In particular,
the `team_id` column is used to uniquely identify teams throughout the software 
(these are the ids used by [sports-reference.com](http://sports-reference.com)).
2. `ap_rankings.csv` - weekly AP top 25 rankings since 2010-11 season. `date` is the date on which the rankings are published
(preseason rankings given a date of 10/1, and postseason a date of 5/1). There are then 25 columns for each 
of the 25 spots in the rankings. Each entry is a list of strings, since it is possible for multiple teams to tie
for a ranking. For example if two teams tie for #4, the `4` columns will have a length-2 list while the `5` column
will have an empty list.
3. `name_substitutions.json` - hardcoded mapping used when trying to decode Reddit game thread titles
4. `game_data` - directory containing game-by-game stats as scraped from [sports-reference.com](http://sports-reference.com).
Not committed to git because they are too large.
5. `gamethreads` - directory containing submission-level information on scraped reddit game threads.
6. `gamethread_comments` - detailed comments from every scraped gamethread
