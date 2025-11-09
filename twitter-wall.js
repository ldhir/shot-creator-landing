// Twitter Wall Management
// This file allows you to manually curate and display tweets with #rooted

class TwitterWall {
    constructor() {
        this.container = document.getElementById('twitter-wall');
        this.tweets = [];
        this.init();
    }

    init() {
        // Load curated tweets
        this.loadCuratedTweets();
        this.renderTweets();
    }

    // Add your curated tweets here
    loadCuratedTweets() {
        // Example format - replace with actual tweets you want to feature
        this.tweets = [
            {
                id: '1',
                author: {
                    name: 'John Smith',
                    handle: '@johnsmith',
                    avatar: 'https://via.placeholder.com/48' // Replace with actual avatar URL
                },
                content: 'Just got 94% similarity to Steph Curry\'s shot! 🏀 This AI analysis is incredible! #rooted',
                date: '2 hours ago',
                link: 'https://twitter.com/johnsmith/status/123456' // Actual tweet link
            },
            {
                id: '2',
                author: {
                    name: 'Sarah Johnson',
                    handle: '@sarahj',
                    avatar: 'https://via.placeholder.com/48'
                },
                content: 'My shooting form is 89% similar to Klay Thompson! Time to work on that last 11% 💪 #rooted',
                date: '5 hours ago',
                link: 'https://twitter.com/sarahj/status/123457'
            },
            {
                id: '3',
                author: {
                    name: 'Mike Davis',
                    handle: '@mikedavis',
                    avatar: 'https://via.placeholder.com/48'
                },
                content: 'This shot analysis tool is a game changer! Got my similarity score and now I know exactly what to improve. #rooted 🎯',
                date: '1 day ago',
                link: 'https://twitter.com/mikedavis/status/123458'
            }
        ];
    }

    renderTweets() {
        if (!this.container) return;

        // Clear placeholder
        this.container.innerHTML = '';

        if (this.tweets.length === 0) {
            this.container.innerHTML = `
                <div class="tweet-placeholder">
                    <p>🐦 No tweets yet!</p>
                    <p class="tweet-instructions">Share your similarity score on Twitter with <strong>#rooted</strong> to be featured here!</p>
                </div>
            `;
            return;
        }

        // Render each tweet
        this.tweets.forEach(tweet => {
            const tweetCard = this.createTweetCard(tweet);
            this.container.appendChild(tweetCard);
        });
    }

    createTweetCard(tweet) {
        const card = document.createElement('div');
        card.className = 'tweet-card';

        // Process content to highlight hashtags
        const processedContent = this.highlightHashtags(tweet.content);

        card.innerHTML = `
            <div class="tweet-header">
                <img src="${tweet.author.avatar}" alt="${tweet.author.name}" class="tweet-avatar" onerror="this.style.display='none'">
                <div class="tweet-author-info">
                    <div class="tweet-author-name">${tweet.author.name}</div>
                    <div class="tweet-handle">${tweet.author.handle}</div>
                </div>
            </div>
            <div class="tweet-content">${processedContent}</div>
            <div class="tweet-footer">
                <span class="tweet-date">${tweet.date}</span>
                <a href="${tweet.link}" target="_blank" class="tweet-link">View on Twitter →</a>
            </div>
        `;

        return card;
    }

    highlightHashtags(content) {
        // Replace #hashtags with styled spans
        return content.replace(/#(\w+)/g, '<span class="tweet-hashtag">#$1</span>');
    }

    // Method to add a new curated tweet
    addTweet(tweet) {
        this.tweets.unshift(tweet); // Add to beginning
        this.renderTweets();
    }

    // Method to remove a tweet by id
    removeTweet(tweetId) {
        this.tweets = this.tweets.filter(t => t.id !== tweetId);
        this.renderTweets();
    }
}

// Initialize Twitter Wall when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    window.twitterWall = new TwitterWall();
});

/*
===========================================
HOW TO ADD NEW CURATED TWEETS:
===========================================

1. Find tweets with #rooted hashtag on Twitter
2. Copy the tweet details
3. Add to the tweets array in loadCuratedTweets() method above

Example format:
{
    id: 'unique-id',
    author: {
        name: 'Full Name',
        handle: '@username',
        avatar: 'https://pbs.twimg.com/profile_images/...' // Twitter avatar URL
    },
    content: 'The tweet text with #rooted hashtag',
    date: '2 hours ago', // Relative time
    link: 'https://twitter.com/username/status/1234567890' // Actual tweet URL
}

OR use the browser console:
window.twitterWall.addTweet({
    id: '4',
    author: { name: 'Jane Doe', handle: '@janedoe', avatar: 'url' },
    content: 'My tweet with #rooted',
    date: 'Just now',
    link: 'https://twitter.com/...'
});

To remove a tweet:
window.twitterWall.removeTweet('tweet-id');

===========================================
ALTERNATIVE: AUTOMATED SOLUTION
===========================================

For a fully automated Twitter wall, you'll need:

1. Twitter API v2 Access:
   - Apply for developer account: https://developer.twitter.com/
   - Create an app and get API credentials
   - Use Twitter API v2 to search for #rooted tweets

2. Backend Server:
   - Create a Node.js/Python backend
   - Use Twitter API to fetch recent tweets with #rooted
   - Cache and serve tweets to your frontend
   - Implement moderation/filtering

3. Third-party Services:
   - Taggbox (https://taggbox.com/) - Paid service
   - Walls.io (https://walls.io/) - Paid service
   - Tweetsmash (https://tweetsmash.com/) - Free with limits

For now, manually curating tweets gives you full control
and ensures only high-quality content is displayed!
*/
