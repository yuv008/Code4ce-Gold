import React from 'react';
import { useLocation } from 'react-router-dom';
import LatestCard from '../components/LatestCard/LatestCard'; // Import your LatestCard component

const SearchResult = () => {
  const location = useLocation();
  const { articles } = location.state || { articles: [] }; // Retrieve articles from state

  return (
    <div className="latestsearch">
      <h2>Search Results</h2>
      <div className="card-container">
        {articles.length > 0 ? (
          articles.map((newsItem, index) => {
            // Check if the title is not empty before rendering the LatestCard
            if (newsItem.title) {
              return (
                <LatestCard
                  key={index}
                  title={newsItem.title}
                  description={newsItem.summary || newsItem.description}
                  imageUrl={newsItem.image || newsItem.urlToImage || "https://salonlfc.com/wp-content/uploads/2018/01/image-not-found-1-scaled.png"}
                  newsItem={newsItem}
                />
              );
            }
            return null; // If the title is empty, return null to not render anything
          })
        ) : (
          <p>No results found.</p> // Message if no articles found
        )}
      </div>
    </div>
  );
}

export default SearchResult;
