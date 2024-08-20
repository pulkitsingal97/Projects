#include <algorithm>
#include <random>
#include <iostream>
#include <string>
#include <cstring>
#include <limits.h>
#include <cmath>
#include <tuple>
#include <vector>

#include "board.hpp"
#include "engine.hpp"
#include "butils.hpp"

using namespace std;

// functions

// butils.hpp
// string move_to_str(U16 move);
// U16 str_to_move(std::string move);
// string show_moves(const BoardData *b, std::unordered_set<U16> &moves);
// string all_boards_to_str(const Board &b);
// char piece_to_char(U8 piece);
// string board_to_str(const BoardData *b);
// string board_7_3_to_str(const U8 *b);

// struct BoardData
// BoardData();
// BoardData(BoardType board_type);
// BoardData(const BoardData &source);

// struct Board
// Board();
// Board(BoardType btype);
// Board(BoardData bdata);
// Board(const Board &source);
// unordered_set<U16> get_legal_moves() const;
// bool in_check() const;
// unordered_set<U16> get_pseudolegal_moves() const;
// unordered_set<U16> get_pseudolegal_moves_for_piece(U8 piece_pos) const;
// bool under_threat(U8 piece_pos) const;
// unordered_set<U16> get_pseudolegal_moves_for_side(U8 color) const;
// void do_move_(U16 move);
// void flip_player_();
// void do_move_without_flip_(U16 move);
// void undo_last_move_without_flip_(U16 move);

int white_manhat_dist(string loc, BoardType btype)
{
    int row = int(loc[1]) - 48;
    char column = loc[0];
    if (btype == SEVEN_THREE)
    {
        if (row <= 4)
            return 5 + (4 - row) + abs(int('b' - column));
        else
            return (6 - row) + abs(int('e' - column));
    }
    else if (btype == EIGHT_FOUR)
    {
        if (row <= 5)
            return 6 + (5 - row) + abs(int('b' - column));
        else
            return (7 - row) + abs(int('f' - column));
    }
    else if (btype == EIGHT_TWO)
    {
        if (row <= 5)
            return 4 + (5 - row) + abs(int('c' - column));
        else
            return (6 - row) + abs(int('f' - column));
    }
}

int black_manhat_dist(string loc, BoardType btype)
{
    int row = int(loc[1]) - 48;
    char column = loc[0];
    if (btype == SEVEN_THREE)
    {
        if (row >= 4)
            return 5 + (row - 4) + abs(int('f' - column));
        else
            return (row - 2) + abs(int('c' - column));
    }
    else if (btype == EIGHT_FOUR)
    {
        if (row >= 5)
            return 7 + (row - 5) + abs(int('g' - column));
        else
            return (row - 2) + abs(int('c' - column));
    }
    else if (btype == EIGHT_TWO)
    {
        if (row >= 5)
            return 5 + (row - 5) + abs(int('f' - column));
        else
            return (row - 3) + abs(int('c' - column));
    }
}

int Evaluate(Board board_copy)
{
    BoardData *board_copy_data = &board_copy.data;
    string board_copy_str = board_to_str(board_copy_data);

    int p = 0, r = 0, b = 0, k = 0, n = 0;
    int P = 0, R = 0, B = 0, K = 0, N = 0;

    int WHITE_evaluation;

    for (int i = 0; i < board_copy_str.size(); i++)
    {
        if (board_copy_str[i] == '\n' || board_copy_str[i] == ' ' || board_copy_str[i] == '.')
            continue;

        switch (board_copy_str[i])
        {
        case 'p':
            p++;
            break;
        case 'r':
            r++;
            break;
        case 'b':
            b++;
            break;
        case 'k':
            k++;
            break;
        case 'n':
            n++;
            break;
        case 'P':
            P++;
            break;
        case 'R':
            R++;
            break;
        case 'B':
            B++;
            break;
        case 'K':
            K++;
            break;
        case 'N':
            N++;
            break;
        }
    }

    WHITE_evaluation = 5 * (B - b) + 3 * (R - r) + 3 * (N - n) + 1 * (P - p);
    return WHITE_evaluation;
}

int Minimax(Board board, int depth, int alpha, int beta, bool MaximizingPlayer)
{
    int maxEval, minEval, eval;

    auto moveset = board.get_legal_moves();
    if (depth == 0 || moveset.size() == 0)
    {
        return Evaluate(board);
    }

    if (MaximizingPlayer)
    {
        maxEval = INT_MIN;
        // cout << "Child Moves: ";
        for (auto m : moveset)
        {
            // cout << move_to_str(m) << "  ";
            Board board_copy(board);
            board_copy.do_move_(m);
            eval = Minimax(board_copy, depth - 1, alpha, beta, false);
            maxEval = max(maxEval, eval);
            alpha = max(alpha, eval);
            if (beta <= alpha)
                break;
        }
        // cout << endl;
        return maxEval;
    }
    else
    {
        minEval = INT_MAX;
        // cout << "Child Moves: ";
        for (auto m : moveset)
        {
            // cout << move_to_str(m) << "  ";
            Board board_copy(board);
            board_copy.do_move_(m);
            eval = Minimax(board_copy, depth - 1, alpha, beta, true);
            minEval = min(minEval, eval);
            beta = min(beta, eval);
            if (beta <= alpha)
                break;
        }
        // cout << endl;
        return minEval;
    }
}

void Engine::find_best_move(const Board &b)
{
    int depth = 2;
    // if (time_left.count() <= 10000)
    // {
    //     depth = 2;
    // }

    auto moveset = b.get_legal_moves();

    if (moveset.size() == 0)
    {
        cout << "Could not get any moves from board!\n";
        cout << board_to_str(&b.data);
        this->best_move = 0;
    }

    // else if (time_left.count() <= 2000)
    // {
    //     vector<U16> moves;
    //     sample(
    //         moveset.begin(),
    //         moveset.end(),
    //         back_inserter(moves),
    //         1,
    //         mt19937{random_device{}()});
    //     this->best_move = moves[0];
    // }

    else
    {
        // cout << "All Legal Moves: ";
        // for (auto m : moveset)
        // {
        //     cout << move_to_str(m) << " ";
        // }
        // cout << endl
        //      << endl;

        int curr_board_evaluation = Evaluate(b);
        int child_board_evaluation;
        int best_child_board_evaluation;

        bool check_mate = false;

        if (b.data.player_to_play == WHITE)
        {
            best_child_board_evaluation = INT_MIN;
            vector<pair<U16, int>> pawn_moveset;
            vector<pair<U16, int>> opp_king_check_moveset;

            for (auto m : moveset)
            {
                // cout << "Move: " << move_to_str(m) << endl;

                Board board_copy(b);
                board_copy.do_move_(m);

                child_board_evaluation = Minimax(board_copy, depth - 1, INT_MIN, INT_MAX, false);
                // cout << "Child Board Evaluation: " << child_board_evaluation << endl;

                if (board_copy.in_check())
                {
                    auto opp_moveset = board_copy.get_legal_moves();
                    if (opp_moveset.size() == 0)
                    {
                        // cout << "*****checkmate_move*****" << endl;
                        this->best_move = m;
                        check_mate = true;
                        break;
                    }

                    // board_copy.flip_player_();
                    // U8 p1 = getp1(m);
                    // if (!(board_copy.under_threat(p1)))
                    // {
                    // cout << "*****opp_king_check_move*****" << endl;
                    opp_king_check_moveset.push_back(make_pair(m, child_board_evaluation));
                    // }
                    // board_copy.flip_player_();
                }

                if (child_board_evaluation > best_child_board_evaluation)
                {
                    best_child_board_evaluation = child_board_evaluation;
                    this->best_move = m;
                }
                // cout << "Best Child Board Evaluation: " << best_child_board_evaluation << endl;

                U8 p0 = getp0(m);
                U8 piece_id = b.data.board_0[p0];
                char piece_name = piece_to_char(piece_id);
                if (piece_name == 'P')
                {
                    // cout << "*****pawn_move*****" << endl;
                    pawn_moveset.push_back(make_pair(m, child_board_evaluation));
                }
                // cout << endl;
            }

            if (!check_mate)
            {
                bool opp_king_check_move = false;

                if (opp_king_check_moveset.size() != 0)
                {
                    if (best_child_board_evaluation >= curr_board_evaluation)
                    {
                        int best_okcm_child_eval = INT_MIN;

                        for (auto okcm : opp_king_check_moveset)
                        {
                            int okcm_child_eval = okcm.second;

                            if (okcm_child_eval >= curr_board_evaluation)
                            {
                                U16 move = okcm.first;

                                // cout << "okcm_move: " << move_to_str(move) << endl;
                                // cout << "okcm_child_eval: " << okcm_child_eval << endl;

                                if (okcm_child_eval > best_okcm_child_eval)
                                {
                                    best_okcm_child_eval = okcm_child_eval;
                                    this->best_move = move;
                                    opp_king_check_move = true;
                                }

                                // cout << "best_okcm_child_eval: " << best_okcm_child_eval << endl
                                //      << endl;
                            }
                        }
                    }
                }

                if (!opp_king_check_move)
                {
                    if (best_child_board_evaluation == curr_board_evaluation)
                    {
                        if (pawn_moveset.size() != 0)
                        {
                            int curr_manhat_dist;
                            int child_manhat_dist;
                            int best_child_manhat_dist = INT_MAX;
                            BoardType btype = b.data.board_type;

                            for (auto pm : pawn_moveset)
                            {
                                int pm_child_eval = pm.second;

                                if (pm_child_eval == best_child_board_evaluation)
                                {
                                    U16 move = pm.first;
                                    string str_move = move_to_str(move);
                                    // cout << "Pawn Move: " << str_move << endl;
                                    curr_manhat_dist = white_manhat_dist(str_move.substr(0, 2), btype);
                                    // cout << "Current Manhattan Distance: " << curr_manhat_dist << endl;
                                    child_manhat_dist = white_manhat_dist(str_move.substr(2, 2), btype);
                                    // cout << "Child Manhattan Distance: " << child_manhat_dist << endl;
                                    if (child_manhat_dist < curr_manhat_dist)
                                    {
                                        if (child_manhat_dist < best_child_manhat_dist)
                                        {
                                            best_child_manhat_dist = child_manhat_dist;
                                            this->best_move = move;
                                        }
                                    }
                                    // cout << "Best Child Manhattan Distance: " << best_child_manhat_dist << endl
                                    //      << endl;
                                }
                            }
                        }
                    }
                }
            }
        }

        else if (b.data.player_to_play == BLACK)
        {
            best_child_board_evaluation = INT_MAX;
            vector<pair<U16, int>> pawn_moveset;
            vector<pair<U16, int>> opp_king_check_moveset;

            for (auto m : moveset)
            {
                // cout << "Move: " << move_to_str(m) << endl;
                Board board_copy(b);
                board_copy.do_move_(m);

                child_board_evaluation = Minimax(board_copy, depth - 1, INT_MIN, INT_MAX, true);
                // cout << "Child Board Evaluation: " << child_board_evaluation << endl;

                if (board_copy.in_check())
                {
                    auto opp_moveset = board_copy.get_legal_moves();
                    if (opp_moveset.size() == 0)
                    {
                        // cout << "*****checkmate_move*****" << endl;
                        this->best_move = m;
                        check_mate = true;
                        break;
                    }

                    // board_copy.flip_player_();
                    // U8 p1 = getp1(m);
                    // if (!(board_copy.under_threat(p1)))
                    // {
                    // cout << "*****opp_king_check_move*****" << endl;
                    opp_king_check_moveset.push_back(make_pair(m, child_board_evaluation));
                    // }
                    // board_copy.flip_player_();
                }

                if (child_board_evaluation < best_child_board_evaluation)
                {
                    best_child_board_evaluation = child_board_evaluation;
                    this->best_move = m;
                }
                // cout << "Best Child_Board Evaluation: " << best_child_board_evaluation << endl;

                U8 p0 = getp0(m);
                U8 piece_id = b.data.board_0[p0];
                char piece_name = piece_to_char(piece_id);
                if (piece_name == 'p')
                {
                    // cout << "*****pawn_move*****" << endl;
                    pawn_moveset.push_back(make_pair(m, child_board_evaluation));
                }

                // cout << endl;
            }

            if (!check_mate)
            {
                bool opp_king_check_move = false;

                if (opp_king_check_moveset.size() != 0)
                {
                    if (best_child_board_evaluation <= curr_board_evaluation)
                    {
                        int best_okcm_child_eval = INT_MAX;

                        for (auto okcm : opp_king_check_moveset)
                        {
                            int okcm_child_eval = okcm.second;

                            if (okcm_child_eval <= curr_board_evaluation)
                            {
                                U16 move = okcm.first;

                                // cout << "okcm_move: " << move_to_str(move) << endl;
                                // cout << "okcm_child_eval: " << okcm_child_eval << endl;

                                if (okcm_child_eval < best_okcm_child_eval)
                                {
                                    best_okcm_child_eval = okcm_child_eval;
                                    this->best_move = move;
                                    opp_king_check_move = true;
                                }

                                // cout << "best_okcm_child_eval: " << best_okcm_child_eval << endl
                                //      << endl;
                            }
                        }
                    }
                }

                if (!opp_king_check_move)
                {
                    if (best_child_board_evaluation == curr_board_evaluation)
                    {
                        if (pawn_moveset.size() != 0)
                        {
                            int curr_manhat_dist;
                            int child_manhat_dist;
                            int best_child_manhat_dist = INT_MAX;
                            BoardType btype = b.data.board_type;

                            for (auto pm : pawn_moveset)
                            {
                                int pm_child_eval = pm.second;

                                if (pm_child_eval == best_child_board_evaluation)
                                {
                                    U16 move = pm.first;
                                    string str_move = move_to_str(move);
                                    // cout << "Pawn Move: " << str_move << endl;
                                    curr_manhat_dist = black_manhat_dist(str_move.substr(0, 2), btype);
                                    // cout << "Current Manhattan Distance: " << curr_manhat_dist << endl;
                                    child_manhat_dist = black_manhat_dist(str_move.substr(2, 2), btype);
                                    // cout << "Child Manhattan Distance: " << child_manhat_dist << endl;
                                    if (child_manhat_dist < curr_manhat_dist)
                                    {
                                        if (child_manhat_dist < best_child_manhat_dist)
                                        {
                                            best_child_manhat_dist = child_manhat_dist;
                                            this->best_move = move;
                                        }
                                    }
                                    // cout << "Best Child Manhattan Distance: " << best_child_manhat_dist << endl
                                    //      << endl;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}