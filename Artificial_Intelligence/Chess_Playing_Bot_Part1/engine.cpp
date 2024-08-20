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

using namespace std;

int white_manhat_dist(string loc)
{
    int row = int(loc[1]) - 48;
    char column = loc[0];
    if (row <= 4)
        return 5 + (4 - row) + min(abs(int('b' - column)), abs(int('f' - column)));
    else
        return (7 - row) + abs(int('d' - column));
}

int black_manhat_dist(string loc)
{
    int row = int(loc[1]) - 48;
    char column = loc[0];
    if (row >= 4)
        return 5 + (row - 4) + min(abs(int('b' - column)), abs(int('f' - column)));
    else
        return (row - 1) + abs(int('d' - column));
}

pair<int, int> Evaluate(Board *board_copy)
{
    U8 *board = board_copy->data.board_0;
    string board_copy_str = board_to_str(board);

    int p = 0, r = 0, b = 0, k = 0;
    int P = 0, R = 0, B = 0, K = 0;

    int WHITE_evaluation;

    for (int i = 0; i < 56; i++)
    {
        // cout << board_copy_str[i];

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
        }
    }
    // cout << B << b << R << r << P << p << endl;

    WHITE_evaluation = 5 * (B - b) + 3 * (R - r) + 1 * (P - p);
    // cout << "Evaluation: " << WHITE_evaluation << endl;

    int num_opp_pieces;
    if (board_copy->data.player_to_play == WHITE)
        num_opp_pieces = p + r + b + k;
    else
        num_opp_pieces = P + R + B + K;

    // cout << "Number of Opponent Pieces: " << num_opp_pieces << endl;

    return make_pair(WHITE_evaluation, num_opp_pieces);
}

int Minimax(Board *board, int depth, int alpha, int beta, bool MaximizingPlayer)
{
    int maxEval, minEval, eval;

    auto moveset = board->get_legal_moves();
    if (depth == 0 || moveset.size() == 0)
    {
        return Evaluate(board).first;
    }

    if (MaximizingPlayer)
    {
        maxEval = INT_MIN;
        for (auto m : moveset)
        {
            // cout << "Child Move: " << move_to_str(m) << "  ";
            Board *board_copy = board->copy();
            board_copy->do_move(m);
            eval = Minimax(board_copy, depth - 1, alpha, beta, false);
            maxEval = max(maxEval, eval);
            alpha = max(alpha, eval);
            if (beta <= alpha)
                break;
        }
        return maxEval;
    }
    else
    {
        minEval = INT_MAX;
        for (auto m : moveset)
        {
            // cout << "Child Move: " << move_to_str(m) << "  ";
            Board *board_copy = board->copy();
            board_copy->do_move(m);
            eval = Minimax(board_copy, depth - 1, alpha, beta, true);
            minEval = min(minEval, eval);
            beta = min(beta, eval);
            if (beta <= alpha)
                break;
        }
        return minEval;
    }
}

void Engine::find_best_move(const Board &b)
{

    auto moveset = b.get_legal_moves();
    if (moveset.size() == 0)
    {
        this->best_move = 0;
    }
    else
    {
        vector<U16> moves;
        cout << all_boards_to_str(b) << endl;
        for (auto m : moveset)
        {
            cout << move_to_str(m) << " ";
        }
        cout << endl;

        int curr_board_evaluation = Evaluate(b.copy()).first;
        int curr_board_num_opp_pieces = Evaluate(b.copy()).second;
        // cout << endl;
        int child_board_evaluation;
        int best_child_board_evaluation;

        int depth = 2;

        if (b.data.player_to_play == WHITE)
        {
            if (curr_board_num_opp_pieces > 1)
            {
                best_child_board_evaluation = INT_MIN;
                vector<pair<U16, int>> pawn_moveset;

                for (auto m : moveset)
                {
                    // cout << "search: " << search << endl;
                    if (!search)
                        break;

                    // cout << "Move: " << move_to_str(m) << endl;
                    Board *board_copy = b.copy();
                    board_copy->do_move(m);

                    child_board_evaluation = Minimax(board_copy, depth - 1, INT_MIN, INT_MAX, false);
                    // cout << "Child Board Evaluation: " << child_board_evaluation << endl;

                    if (board_copy->in_check())
                    {
                        auto opp_moveset = board_copy->get_legal_moves();
                        if (opp_moveset.size() == 0)
                        {
                            this->best_move = m;
                            break;
                        }
                    }

                    U8 p0 = getp0(m);
                    U8 piece_id = b.data.board_0[p0];
                    char piece_name = piece_to_char(piece_id);
                    // cout << piece_name << endl;
                    if (piece_name == 'P')
                    {
                        pawn_moveset.push_back(make_pair(m, child_board_evaluation));
                    }

                    if (child_board_evaluation > best_child_board_evaluation)
                    {
                        best_child_board_evaluation = child_board_evaluation;
                        this->best_move = m;
                    }
                    // cout << "Best Child Board Evaluation: " << best_child_board_evaluation << endl
                        //  << endl;
                }

                if (best_child_board_evaluation == curr_board_evaluation)
                {
                    if (pawn_moveset.size() != 0)
                    {
                        int curr_manhat_dist;
                        int child_manhat_dist;
                        int best_child_manhat_dist = INT_MAX;

                        for (auto pm : pawn_moveset)
                        {
                            // cout << "search: " << search << endl;
                            if (!search)
                                break;

                            int pm_child_eval = pm.second;

                            if (pm_child_eval == best_child_board_evaluation)
                            {
                                U16 move = pm.first;
                                string str_move = move_to_str(move);
                                // cout << "Pawn Move: " << str_move << endl;
                                curr_manhat_dist = white_manhat_dist(str_move.substr(0, 2));
                                // cout << "Current Manhattan Distance: " << curr_manhat_dist << endl;
                                child_manhat_dist = white_manhat_dist(str_move.substr(2, 2));
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
                                    //  << endl;
                            }
                        }
                    }
                }
            }
            else
            {
                best_child_board_evaluation = INT_MIN;
                vector<pair<U16, int>> one_king_moveset;

                for (auto m : moveset)
                {
                    // cout << "search: " << search << endl;
                    if (!search)
                        break;

                    // cout << "Move: " << move_to_str(m) << endl;
                    Board *board_copy = b.copy();
                    board_copy->do_move(m);

                    child_board_evaluation = Minimax(board_copy, depth - 1, INT_MIN, INT_MAX, false);
                    // cout << "Child Board Evaluation: " << child_board_evaluation << endl;

                    auto opp_moveset = board_copy->get_legal_moves();
                    if (opp_moveset.size() == 0)
                    {
                        this->best_move = m;
                        break;
                    }

                    one_king_moveset.push_back(make_pair(m, child_board_evaluation));

                    if (child_board_evaluation > best_child_board_evaluation)
                    {
                        best_child_board_evaluation = child_board_evaluation;
                        this->best_move = m;
                    }
                    // cout << "Best Child Board Evaluation: " << best_child_board_evaluation << endl
                        //  << endl;
                }

                if (best_child_board_evaluation == curr_board_evaluation)
                {
                    int best_num_opp_moves = INT_MAX;

                    for (auto okm : one_king_moveset)
                    {
                        // cout << "search: " << search << endl;
                        if (!search)
                            break;

                        int okm_child_eval = okm.second;

                        if (okm_child_eval == best_child_board_evaluation)
                        {
                            U16 move = okm.first;
                            string str_move = move_to_str(move);
                            // cout << "One King Move: " << str_move << endl;
                            Board *board_copy = b.copy();
                            board_copy->do_move(move);
                            auto opp_moveset2 = board_copy->get_legal_moves();
                            int num_opp_moves = opp_moveset2.size();
                            // cout << "Number of Opponent Moves: " << num_opp_moves << endl;
                            if (num_opp_moves < best_num_opp_moves)
                            {
                                best_num_opp_moves = num_opp_moves;
                                this->best_move = move;
                            }
                            // cout << "Best Number of Opponent Moves: " << best_num_opp_moves << endl;
                        }
                    }
                }
            }
        }

        else if (b.data.player_to_play == BLACK)
        {
            if (curr_board_num_opp_pieces > 1)
            {
                best_child_board_evaluation = INT_MAX;
                vector<pair<U16, int>> pawn_moveset;

                for (auto m : moveset)
                {
                    // cout << "search: " << search << endl;
                    if (!search)
                        break;

                    // cout << "Move: " << move_to_str(m) << endl;
                    Board *board_copy = b.copy();
                    board_copy->do_move(m);

                    child_board_evaluation = Minimax(board_copy, depth - 1, INT_MIN, INT_MAX, true);
                    // cout << "Child Board Evaluation: " << child_board_evaluation << endl;

                    if (board_copy->in_check())
                    {
                        auto opp_moveset = board_copy->get_legal_moves();
                        if (opp_moveset.size() == 0)
                        {
                            this->best_move = m;
                            break;
                        }
                    }

                    U8 p0 = getp0(m);
                    U8 piece_id = b.data.board_0[p0];
                    char piece_name = piece_to_char(piece_id);
                    // cout << piece_name << endl;
                    if (piece_name == 'p')
                    {
                        pawn_moveset.push_back(make_pair(m, child_board_evaluation));
                    }

                    if (child_board_evaluation < best_child_board_evaluation)
                    {
                        best_child_board_evaluation = child_board_evaluation;
                        this->best_move = m;
                    }
                    // cout << "Best Child_Board Evaluation: " << best_child_board_evaluation << endl
                        //  << endl;
                }

                if (best_child_board_evaluation == curr_board_evaluation)
                {
                    if (pawn_moveset.size() != 0)
                    {
                        int curr_manhat_dist;
                        int child_manhat_dist;
                        int best_child_manhat_dist = INT_MAX;

                        for (auto pm : pawn_moveset)
                        {
                            // cout << "search: " << search << endl;
                            if (!search)
                                break;

                            int pm_child_eval = pm.second;

                            if (pm_child_eval == best_child_board_evaluation)
                            {
                                U16 move = pm.first;
                                string str_move = move_to_str(move);
                                // cout << "Pawn Move: " << str_move << endl;
                                curr_manhat_dist = black_manhat_dist(str_move.substr(0, 2));
                                // cout << "Current Manhattan Distance: " << curr_manhat_dist << endl;
                                child_manhat_dist = black_manhat_dist(str_move.substr(2, 2));
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
                                    //  << endl;
                            }
                        }
                    }
                }
            }
            else
            {
                best_child_board_evaluation = INT_MAX;
                vector<pair<U16, int>> one_king_moveset;

                for (auto m : moveset)
                {
                    // cout << "search: " << search << endl;
                    if (!search)
                        break;

                    // cout << "Move: " << move_to_str(m) << endl;
                    Board *board_copy = b.copy();
                    board_copy->do_move(m);

                    child_board_evaluation = Minimax(board_copy, depth - 1, INT_MIN, INT_MAX, true);
                    // cout << "Child Board Evaluation: " << child_board_evaluation << endl;

                    auto opp_moveset = board_copy->get_legal_moves();
                    if (opp_moveset.size() == 0)
                    {
                        this->best_move = m;
                        break;
                    }

                    one_king_moveset.push_back(make_pair(m, child_board_evaluation));

                    if (child_board_evaluation < best_child_board_evaluation)
                    {
                        best_child_board_evaluation = child_board_evaluation;
                        this->best_move = m;
                    }
                    // cout << "Best Child Board Evaluation: " << best_child_board_evaluation << endl
                        //  << endl;
                }

                if (best_child_board_evaluation == curr_board_evaluation)
                {
                    int best_num_opp_moves = INT_MAX;

                    for (auto okm : one_king_moveset)
                    {
                        // cout << "search: " << search << endl;
                        if (!search)
                            break;

                        int okm_child_eval = okm.second;

                        if (okm_child_eval == best_child_board_evaluation)
                        {
                            U16 move = okm.first;
                            string str_move = move_to_str(move);
                            // cout << "One King Move: " << str_move << endl;
                            Board *board_copy = b.copy();
                            board_copy->do_move(move);
                            auto opp_moveset2 = board_copy->get_legal_moves();
                            int num_opp_moves = opp_moveset2.size();
                            // cout << "Number of Opponent Moves: " << num_opp_moves << endl;
                            if (num_opp_moves < best_num_opp_moves)
                            {
                                best_num_opp_moves = num_opp_moves;
                                this->best_move = move;
                            }
                            // cout << "Best Number of Opponent Moves: " << best_num_opp_moves << endl
                                //  << endl;
                        }
                    }
                }
            }
        }

        // pick a random move
        // std::sample(
        //     moveset.begin(),
        //     moveset.end(),
        //     std::back_inserter(moves),
        //     1,
        //     std::mt19937{std::random_device{}()}
        // );
        // this->best_move = moves[0];
    }
}