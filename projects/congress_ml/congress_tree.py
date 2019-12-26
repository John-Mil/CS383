"""
John Milmore
==============================================
Train a decision tree on congress voting data
==============================================
"""

import train_tree

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

FILE_NAME = 'congress_data.csv'


def clean_votes(votes):
    """Clean votes in place.

        Arguments:
            votes -- array-like, shape (num_voters, num_votes)

        Returns:
            None. Alters votes in place such that no missing entries. All entries are 1 - Yea/Aye or 0 - No/Nay.
            The non voting entries are first converted to missing entries (-1) and then these entries are replaced by
            the majority response for the corresponding vote.
    """
    num_voters = len(votes)
    for vote_j in votes.T:
        for i in range(num_voters):
            # Convert to ints
            if vote_j[i] in {'Yea', 'Aye'}:
                vote_j[i] = 1
            elif vote_j[i] in {'No', 'Nay'}:
                vote_j[i] = 0
            else:
                vote_j[i] = -1
        # Replace missing values with majority votes
        yea_mask = vote_j == 1
        no_mask = vote_j == 0
        none_mask = vote_j == -1
        if sum(yea_mask) > sum(no_mask):
            vote_j[none_mask] = 1
        else:
            vote_j[none_mask] = 0


def accuracy(t, X, y_true):
    """Return accuracy of decision tree t.

        Arguments:
            t -- Node (train_tree)
                root node of trained decision tree
            X -- array-like
                voting records to predict party
            y -- array-like
                actual parties of voters

        Returns:
            accuracy -- int
    """
    y_pred = [t.predict(x) for x in X]
    return sum(y_pred == y_true) / len(y_true)


def main():

    # #########################################################################
    # Get voting data

    votes_df = pd.read_csv(FILE_NAME).drop(columns=['Name', 'State', 'District'])
    votes = votes_df.drop(columns=['Party']).values
    party = votes_df['Party'].values

    # Clean voting data
    clean_votes(votes)

    # #########################################################################
    # Split data into train and test and train decision tree

    X_train, X_test, y_train, y_test = train_test_split(votes, party, test_size=0.25)

    # Train the tree on a subset of the features (votes)
    t = train_tree.learn_decision_tree(X_train, y_train, range(0, 10), 4)

    # Train accuracy
    print(f'Train accuracy: {accuracy(t, X_train, y_train):.4f}')

    # Test accuracy
    print(f'Test accuracy: {accuracy(t, X_test, y_test):.4f}')

    # #########################################################################
    # Find optimal depth limit
    # Feed all of the features to the decision tree learner. The algorithm chooses the best features to split on. This
    # number is capped by the depth limit.

    train_acc_ls, test_acc_ls = [], []
    depth_range = range(0, 8)
    for d in depth_range:
        t = train_tree.learn_decision_tree(X_train, y_train, range(len(X_train.T)), d)
        train_acc_ls.append(accuracy(t, X_train, y_train))
        test_acc_ls.append(accuracy(t, X_test, y_test))

    fig, ax = plt.subplots(dpi=400)
    sns.set()
    ax.plot(depth_range, train_acc_ls, '.-', label='training')
    ax.plot(depth_range, test_acc_ls, '.-', label='testing')
    ax.legend()
    ax.set_xlabel('Depth limit')
    ax.set_ylabel('Accuracy')
    ax.set_xticks(depth_range)
    ax.set_title('Training a Decision Tree on Congress Data')
    fig.savefig('train_vs_test.png')


if __name__ == '__main__':
    main()
