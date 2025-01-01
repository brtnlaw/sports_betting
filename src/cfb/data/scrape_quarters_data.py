import datetime as dt
def fill_quarters_data(date, home):
    base_url = "https://www.sports-reference.com/cfb/boxscores/index.cgi"
    url = f"{base_url}/{date.strftime('%Y-%m-%d')}-{home}.html"
    html = urlopen(url)
    soup = BeautifulSoup(html, features="html.parser")
    cand_divs = soup.find_all(attrs={"class": re.compile("game_summaries")})
    pass


def backfill_data_for_table(start_date):
    # Starting from reverse date
    # Update the table, update the log
    pass
    
