from enum import StrEnum
import param
import panel as pn
import pandas as pd
import holoviews as hv
import hvplot.pandas
import wordcloud as wc
from collections import Counter
import plotly.express as px
import plotly.io as pio
import seaborn as sns
import matplotlib.pyplot as plt

pio.renderers.default = "notebook_connected"
pn.extension("plotly")
hv.extension("bokeh")


class Duration(StrEnum):
    """An enumeration representing different time durations.

    Attributes:
        YEARLY (str): Represents yearly aggregation.
        MONTHLY (str): Represents monthly aggregation.
        WEEKLY (str): Represents weekly aggregation.
        HOURLY (str): Represents hourly aggregation.
    """

    YEARLY: str = "Yearly"
    MONTHLY: str = "Monthly"
    WEEKLY: str = "Weekly"
    HOURLY: str = "Hourly"

    @classmethod
    def list(cls):
        """Generates a list of all duration values.

        Returns:
            list: A list of string values representing all durations.
        """
        return list(map(lambda c: c.value, cls))


def chat_activity_curve(
    df: pd.DataFrame, duration: Duration = Duration.YEARLY
):
    """Generates a curve plot of chat activity over a specified duration.

    Args:
        df (pd.DataFrame): The input dataframe containing chat activity data.
        duration (Duration): The time duration for grouping data. Defaults to YEARLY.

    Returns:
        hvplot: A Holoviews plot showing chat activity over the specified duration.
    """
    if duration == Duration.HOURLY:
        return (
            df.groupby("hour")
            .size()
            .reset_index(name="count")
            .hvplot("hour", "count")
        )
    elif duration == Duration.WEEKLY:
        return (
            df.groupby(["day", "day_name"])
            .size()
            .reset_index(name="count")
            .hvplot("day_name", "count")
        )
    elif duration == Duration.MONTHLY:
        return (
            df[df.year == 2021]
            .groupby(["month", "month_name"])
            .size()
            .reset_index(name="count")
            .hvplot("month_name", "count")
        ) * (
            df[df.year == 2020]
            .groupby(["month", "month_name"])
            .size()
            .reset_index(name="count")
            .hvplot("month_name", "count")
        )
    else:
        return (
            df.groupby(df["date"].dt.date)
            .size()
            .reset_index(name="count")
            .hvplot("date", "count")
        )


class ValochatDashboard(param.Parameterized):
    """A dashboard for visualizing chat activity, word clouds, and user data.

    Attributes:
        data (pd.DataFrame): The chat activity dataset.
        duration_select (pn.widgets.Select): Dropdown for selecting duration (e.g., Hourly, Weekly).
        words_select (pn.widgets.Select): Dropdown for selecting word categories.
    """

    def __init__(self, df, **params):
        """Initializes the ValochatDashboard with the dataset and parameters.

        Args:
            df (pd.DataFrame): The input chat activity dataset.
            **params: Additional parameters for Panel widgets.
        """
        self.data = df

        self.duration_select = pn.widgets.Select(
            name="Duration", options=Duration.list()
        )
        self.words_select = pn.widgets.Select(
            name="Words",
            options=["All Terms", "General Terms", "In-Game Terms"],
            value="All Terms",
        )

        super().__init__(**params)

        self.tab1 = ("Introduction", self.introduction)

        self.tab2 = (
            "Chat Activity",
            pn.Column(
                self.duration_select,
                self.chat_activity_curve,
            ),
        )

        self.tab3 = (
            "Word Cloud",
            pn.Column(self.words_select, self.wordcloud),
        )

        self.tab4 = ("3D Plot", self.users3dPlot)

        self.tab5 = ("Heat Map", self.heatmap)

        self.tab6 = ("Top Users", self.topusers)

        self.tab7 = ("Retention Map", self.retention_map)

        self.layout = pn.Tabs(
            self.tab1,
            self.tab2,
            self.tab3,
            self.tab4,
            self.tab5,
            self.tab6,
            self.tab7,
        )

    @property
    def introduction(self):
        """Generates the introduction panel.

        Returns:
            pn.Column: A Panel column with project title and description.
        """
        title = "# Valochat"
        text = """
        We will be exploring chat activity of gamers
        in general channel of valorant discord server.
        """
        return pn.Column(
            title,
            pn.layout.Divider(),
            text,
        )

    @param.depends("duration_select.value")
    def chat_activity_curve(self):
        """Generates a curve plot for chat activity based on the selected duration.

        Returns:
            hvplot: A curve plot of chat activity.
        """
        # print(self.duration_select.value)
        return chat_activity_curve(self.data, self.duration_select.value)

    @param.depends("words_select.value")
    def wordcloud(self):
        """Generates a word cloud based on selected word categories.

        Returns:
            pn.pane.Image: A Panel image displaying the generated word cloud.
        """
        game_terms = list(
            map(
                lambda t: t.lower(),
                (
                    "Agent,Spike,Plant,Defuse,Phantom,Vandal,Jett,Reyna,Sage,Cypher,"
                    + "Raze,Killjoy,Breach,Omen,Brimstone,Astra,Yoru,Chamber,KAYO,Harbor,"
                    + "Fracture,Split,Bind,Ascent,Pearl,Lotus,Breeze,Haven,Valorant,"
                    + "Ranked,Unrated,Matchmaking,Operator,Guardian,Sheriff,Classic,Bulldog,"
                    + "Marshal,Shorty,Stinger,Spectre,Ghost,Tactical,Shooter,Clutch,"
                    + "Ace,Flawless,Economy,Teamwork,Communication,Ultimate,Ability,Flash,"
                    + "Smoke,Molotov,Recon,Knife,Headhunter,Teleport,Resurrect,Healing,Shield"
                ).split(","),
            )
        )

        text = " ".join(self.data["content"].tolist())

        stopwords = set(wc.STOPWORDS)
        stopwords.update(["Deleted", "User", "game"])
        if self.words_select.value == "General Terms":
            stopwords.update(game_terms)

        if self.words_select.value == "In-Game Terms":
            word_counts = Counter()
            for word in text.lower().split():
                if word in game_terms:
                    if word in word_counts:
                        word_counts[word] += 1
                    else:
                        word_counts[word] = 1
            wordcloud = wc.WordCloud(
                width=800,
                height=400,
                background_color="black",
                max_words=100,
                contour_color="white",
            ).generate_from_frequencies(word_counts)
        else:
            wordcloud = wc.WordCloud(
                width=800,
                height=400,
                background_color="white",
                stopwords=stopwords,
                min_font_size=10,
                collocations=False,
            ).generate(text)

        return pn.pane.Image(wordcloud.to_image())

    @property
    def users3dPlot(self):
        """Generates a 3D scatter plot showing user activity metrics.

        Returns:
            plotly.graph_objects.Figure: A 3D scatter plot visualizing date, message count, reactions, and user activity.
        """
        df_by_dates = self.data.groupby(self.data["date"].dt.date)
        df_stats = pd.DataFrame(
            {
                "date": df_by_dates["date"].first(),
                "users": df_by_dates["authorid"].nunique(),
                "reactions": df_by_dates["reactions_total"].sum(),
                "messages": df_by_dates.size(),
                "sentiment": df_by_dates["sentiment"].mean(),
            }
        )

        return px.scatter_3d(
            df_stats[df_stats["reactions"] <= 15000],
            x="date",
            y="messages",
            z="reactions",
            size="users",
            color="sentiment",
            size_max=18,
            opacity=0.7,
        )

    @property
    def heatmap(self):
        """Generates a heatmap for user activity (day vs hour) for top users.

        Returns:
            hv.HeatMap: A heatmap showing the activity distribution of the top 10 users.
        """
        daily_message_count = (
            self.data.groupby(["date", "author"])["content"]
            .count()
            .reset_index(name="count")
        )
        top_users = (
            daily_message_count.groupby("author")["count"]
            .sum()
            .nlargest(10)
            .index
        )
        hourly_activity = (
            self.data.groupby(["author", "day", "hour"])["content"]
            .count()
            .reset_index(name="count")
        )
        hourly_activity_top_users = hourly_activity[
            hourly_activity["author"].isin(top_users)
        ]
        heatmap = hv.HeatMap(
            hourly_activity_top_users, kdims=["day", "hour"], vdims="count"
        ).opts(
            colorbar=True,
            cmap="Blues",
            xlabel="Day of Week (0 = Monday)",
            ylabel="Hour of Day",
            width=900,
            height=500,
            title="User Activity Heatmap (Day vs Hour) - Top 10 Users",
        )
        return heatmap

    @property
    def topusers(self):
        """Displays a DataFrame of the most active users.

        Returns:
            pn.widgets.DataFrame: A table showing the top users and their message counts.
        """
        daily_message_count = (
            self.data.groupby(["date", "author"])["content"]
            .count()
            .reset_index(name="count")
        )
        top_users = (
            daily_message_count.groupby("author")["count"]
            .sum()
            .nlargest(11)
            .index
        )
        return pn.widgets.DataFrame(
            self.data[self.data["author"].isin(top_users)]
            .groupby("author")["content"]
            .count()
            .reset_index(name="count")
        )

    @property
    def retention_map(self):
        """Generates a cohort retention heatmap showing user retention over time.

        Returns:
            pn.pane.Matplotlib: A Matplotlib pane displaying the retention heatmap.
        """
        self.data["CohortMonth"] = (
            self.data.groupby("authorid")["date"]
            .transform("min")
            .dt.to_period("M")
        )
        self.data["ActivityMonth"] = self.data["date"].dt.to_period("M")

        cohort_data = (
            self.data.groupby(["CohortMonth", "ActivityMonth"])["authorid"]
            .nunique()
            .reset_index()
        )
        cohort_data["CohortIndex"] = (
            cohort_data["ActivityMonth"] - cohort_data["CohortMonth"]
        ).apply(lambda x: x.n)
        cohort_pivot = cohort_data.pivot(
            index="CohortMonth", columns="CohortIndex", values="authorid"
        )

        cohort_sizes = cohort_pivot.iloc[:, 0]
        retention_matrix = cohort_pivot.divide(cohort_sizes, axis=0) * 100

        fig = plt.Figure(figsize=(9, 6))
        ax = fig.add_subplot(111)
        sns.heatmap(
            retention_matrix,
            ax=ax,
            annot=True,
            fmt=".1f",
            cmap="Blues",
            cbar_kws={"label": "Retention %"},
        )
        return pn.pane.Matplotlib(fig)
