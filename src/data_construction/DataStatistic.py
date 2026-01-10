

class DataStatistic:


    def __init__(self):
        self.prompt_num = 0
        self.news_num_total = 0
        self.max_news_num_per_prompt = 0
        self.min_news_num_per_prompt = 0
        self.mean_news_num_per_prompt = 0.0
        self.the_prompt_with_most_news_num = ""

        self.prompt_with_max_total_len = ""
        self.prompt_with_max_news_len = ""
        self.newslen_prompt_with_max_news_len = 0


    def news_num_stats_update(self, news_num: int, prompt: str):

        self.prompt_num += 1
        self.news_num_total += news_num

        # max and min
        if news_num > self.max_news_num_per_prompt:
            self.max_news_num_per_prompt = news_num
            # maintain the prompt with most number ofnews
            self.the_prompt_with_most_news_num = prompt

        if self.min_news_num_per_prompt == 0 or news_num < self.min_news_num_per_prompt:
            self.min_news_num_per_prompt = news_num
        # mean
        self.mean_news_num_per_prompt = self.news_num_total / self.prompt_num

        # prompt with max total length
        if len(prompt) > len(self.prompt_with_max_total_len):
            self.prompt_with_max_total_len = prompt
        # prompt with max news length
        news_part = prompt.split("News:\n")[-1]
        if len(news_part) > len(self.prompt_with_max_news_len):
            self.prompt_with_max_news_len = news_part
            self.newslen_prompt_with_max_news_len = len(news_part)

    def clear(self):
        self.prompt_num = 0
        self.news_num_total = 0
        self.max_news_num_per_prompt = 0
        self.min_news_num_per_prompt = 0
        self.mean_news_num_per_prompt = 0.0
        self.the_prompt_with_most_news_num = ""

        self.prompt_with_max_total_len = ""
        self.prompt_with_max_news_len = ""