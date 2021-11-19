#ifndef MLIR_TUTORIAL_TOY_PARSER_H
#define MLIR_TUTORIAL_TOY_PARSER_H

#include "Lexer.h"
#include "AST.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <utility>
#include <vector>

namespace toy {

class Parser {
public:
    Parser(Lexer &lexer) : lexer(lexer) {}

    std::unique_ptr<ModuleAST> parseModule() {
        lexer.getNextToken();

        std::vector<FunctionAST> functions;
        while (auto f = parseDefinition())
        {
            functions.push_back(std::move(*f));
            if (lexer.getCurToken() == tok_eof)
                break;
        }

        if (lexer.getCurToken() != tok_eof)
            return parseError<ModuleAST>("nothing", "at end of module");
        
        return std::make_unique<ModuleAST>(std::move(functions));
    }

private:
    Lexer &lexer;

    int getTokPrecedence() {
        if (!isascii(lexer.getCurToken()))
            return -1;
        
        switch (static_cast<char>(lexer.getCurToken()))
        {
        case '-':
            return 20;
        case '+':
            return 20;
        case '*':
            return 40;
        default:
            return -1;
        }
    }

    std::unique_ptr<ReturnExprAST> parseReturn() {
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_return);

        llvm::Optional<std::unique_ptr<ExprAST>> expr;
        if (lexer.getCurToken() != ';') {
            expr = parseExpression();
            if (!expr)
                return nullptr;
        }
        return std::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
    }

    std::unique_ptr<ExprAST> parseNumberExpr() {
        auto loc = lexer.getLastLocation();
        auto result = std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
        lexer.consume(tok_number);
        return std::move(result);
    }

    std::unique_ptr<ExprAST> parseTensorLiteralExpr() {
        auto loc = lexer.getLastLocation();
        lexer.consume(Token('['));

        std::vector<std::unique_ptr<ExprAST>> values;
        std::vector<int64_t> dims;
        do {
            if (lexer.getCurToken() == '[') {
                values.push_back(parseTensorLiteralExpr());
                if (!values.back())
                    return nullptr;
            } else {
                if (lexer.getCurToken() != tok_number)
                    return parseError<ExprAST>("<num> or [", "in literal expression");
                values.push_back(parseNumberExpr());
            }

            if (lexer.getCurToken() == ']')
                break;
            
            if (lexer.getCurToken() != ',')
                return parseError<ExprAST>("] or ,", "in literal expression");
            
            lexer.getNextToken();
        } while(true);
        if (values.empty())
            return parseError<ExprAST>("<something>", "to fill literal expression");
        lexer.getNextToken();

        dims.push_back(values.size());

        if (llvm::any_of(values, [](std::unique_ptr<ExprAST> &expr) {
            return llvm::isa<LiteralExprAST>(expr.get());
        })) {
            auto firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front().get());
            if (!firstLiteral)
                return parseError<ExprAST>("uniform well-nested dimensions",
                                           "inside literal expression");
            
            auto firstDims = firstLiteral->getDims();
            dims.insert(dims.end(), firstDims.begin(), firstDims.end());

            for (auto &expr : values) {
                auto *exprLiteral = llvm::cast<LiteralExprAST>(expr.get());
                if (!exprLiteral)
                    return parseError<ExprAST>("uniform well-nested dimensions",
                                               "inside literal expression");
                if (exprLiteral->getDims() != firstDims)
                    return parseError<ExprAST>("uniform well-nested dimensions",
                                               "inside literal expression");
            }
        }

        return std::make_unique<LiteralExprAST>(std::move(loc), std::move(values), std::move(dims));
    }

    std::unique_ptr<ExprAST> parseParenExpr() {
        lexer.getNextToken();
        auto v = parseExpression();
        if (!v)
            return nullptr;
        
        if (lexer.getCurToken() != ')')
            return parseError<ExprAST>(")", "to close expression with parentheses");
        lexer.consume(Token(')'));
        
        return v;
    }

    std::unique_ptr<ExprAST> parseIdentifierExpr() {
        std::string name(lexer.getId());

        auto loc = lexer.getLastLocation();
        lexer.getNextToken();

        if (lexer.getCurToken() != '(')
            return std::make_unique<VariableExprAST>(std::move(loc), name);

        lexer.consume(Token('('));
        std::vector<std::unique_ptr<ExprAST>> args;
        if (lexer.getCurToken() != ')') {
            while (true) {
                if (auto arg = parseExpression())
                    args.push_back(std::move(arg));
                else
                    return nullptr;
                
                if (lexer.getCurToken() == ')')
                    break;
                
                if (lexer.getCurToken() != ',')
                    return parseError<ExprAST>(", or )", "in argument list");
                lexer.consume(Token(')'));
            }
        }
        lexer.consume(Token(')'));

        if (name == "print") {
            if (args.size() != 1)
                return parseError<ExprAST>("<single arg>", "as argument to print()");
            
            return std::make_unique<PrintExprAST>(std::move(loc), std::move(args[0]));
        }

        return std::make_unique<CallExprAST>(std::move(loc), name, std::move(args));
    }

    std::unique_ptr<ExprAST> parsePrimary() {
        switch (lexer.getCurToken())
        {
        case tok_identifier:
            return parseIdentifierExpr();
        case tok_number:
            return parseNumberExpr();
        case '(':
            return parseParenExpr();
        case '[':
            return parseTensorLiteralExpr();
        case ';':
            return nullptr;
        case '}':
            return nullptr;
        
        default:
            llvm::errs() << "unknown token '" << lexer.getCurToken()
                         << "' when expecting an expression\n";
        }
    }

    std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                           std::unique_ptr<ExprAST> lhs) {
        while (true) {
            int tokPrec = getTokPrecedence();

            if (tokPrec < exprPrec)
                return lhs;
            
            int binOp = lexer.getCurToken();
            lexer.consume(Token(binOp));
            auto loc = lexer.getLastLocation();

            auto rhs = parsePrimary();
            if (!rhs)
                return parseError<ExprAST>("expression", "to complete binary operator");

            int nextPrec = getTokPrecedence();
            if (tokPrec < nextPrec) {
                rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
                if (!rhs)
                    return nullptr;
            }

            lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp,
                                                  std::move(lhs), std::move(rhs));
        }
    }

    std::unique_ptr<ExprAST> parseExpression() {
        auto lhs = parsePrimary();
        if (!lhs)
            return nullptr;

        return parseBinOpRHS(0, std::move(lhs));
    }

    std::unique_ptr<VarType> parseType() {
        if (lexer.getCurToken() != '<')
            return parseError<VarType>("<", "to begin type");
        lexer.getNextToken();

        auto type = std::make_unique<VarType>();

        while (lexer.getCurToken() == tok_number) {
            type->shape.push_back(lexer.getValue());
            lexer.getNextToken();
            if (lexer.getCurToken() == ',')
                lexer.getNextToken();
        }

        if (lexer.getCurToken() != '>')
            return parseError<VarType>(">", "to end type");
        lexer.getNextToken();
        return type;
    }

    std::unique_ptr<VarDeclExprAST> parseDeclaration() {
        if (lexer.getCurToken() != tok_var)
            return parseError<VarDeclExprAST>("var", "to begin declaration");
        auto loc = lexer.getLastLocation();
        lexer.getNextToken();

        if (lexer.getCurToken() != tok_identifier)
            return parseError<VarDeclExprAST>("identified", "after 'var' declaration");
        
        std::string id(lexer.getId());
        lexer.getNextToken();

        std::unique_ptr<VarType> type;
        if (lexer.getCurToken() == '<') {
            type = parseType();
            if (!type)
                return nullptr;
        }

        if (!type)
            type = std::make_unique<VarType>();
        lexer.consume(Token('='));
        auto expr = parseExpression();
        return std::make_unique<VarDeclExprAST>(std::move(loc), std::move(id),
                                                std::move(*type), std::move(expr));
    }

    std::unique_ptr<ExprASTList> parseBlock() {
        if (lexer.getCurToken() != '{')
            return parseError<ExprASTList>("{", "to begin block");
        lexer.consume(Token('{'));

        auto exprList = std::make_unique<ExprASTList>();

        while (lexer.getCurToken() == ';')
            lexer.consume(Token(';'));

        while (lexer.getCurToken() != '}' && lexer.getCurToken() != tok_eof) {
            if (lexer.getCurToken() == tok_var) {
                auto varDecl = parseDeclaration();
                if (!varDecl)
                    return nullptr;
                exprList->push_back(std::move(varDecl));
            } else if (lexer.getCurToken() == tok_return) {
                auto ret = parseReturn();
                if (!ret)
                    return nullptr;
                exprList->push_back(std::move(ret));
            } else {
                auto expr = parseExpression();
                if (!expr)
                    return nullptr;
                exprList->push_back(std::move(expr));
            }

            if (lexer.getCurToken() != ';')
                return parseError<ExprASTList>(";", "after expression");
            
            while (lexer.getCurToken() == ';')
                lexer.consume(Token(';'));
        }

        if (lexer.getCurToken() != '}')
            return parseError<ExprASTList>("}", "to close block");
        
        lexer.consume(Token('}'));
        return exprList;
    }

    std::unique_ptr<PrototypeAST> parsePrototype() {
        auto loc = lexer.getLastLocation();

        if (lexer.getCurToken() != tok_def)
            return parseError<PrototypeAST>("def", "in prototype");
        lexer.consume(tok_def);

        if (lexer.getCurToken() != tok_identifier)
            return parseError<PrototypeAST>("function name", "in prototype");

        std::string fnName(lexer.getId());
        lexer.consume(tok_identifier);

        if (lexer.getCurToken() != '(')
            return parseError<PrototypeAST>("(", "in prototype");
        lexer.consume(Token('('));

        std::vector<std::unique_ptr<VariableExprAST>> args;
        if (lexer.getCurToken() != ')') {
            do {
                std::string name(lexer.getId());
                auto loc = lexer.getLastLocation();
                lexer.consume(tok_identifier);
                auto decl = std::make_unique<VariableExprAST>(std::move(loc), name);
                args.push_back(std::move(decl));
                if (lexer.getCurToken() != ',')
                    break;
                lexer.consume(Token(','));
                if (lexer.getCurToken() != tok_identifier)
                    return parseError<PrototypeAST>(
                        "identifier", "after ',' in function parameter list");
            } while (true);
        }
        if (lexer.getCurToken() != ')')
            return parseError<PrototypeAST>(")", "to end function prototype");
        
        lexer.consume(Token(')'));
        return std::make_unique<PrototypeAST>(std::move(loc), fnName, std::move(args));
    }

    std::unique_ptr<FunctionAST> parseDefinition() {
        auto proto = parsePrototype();
        if (!proto)
            return nullptr;
        
        if (auto block = parseBlock())
            return std::make_unique<FunctionAST>(std::move(proto), std::move(block));
        return nullptr;
    }

    template <typename R, typename T, typename U = const char *>
    std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
        auto curToken = lexer.getCurToken();
        llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ","
                     << lexer.getLastLocation().col << "): expected '" << expected
                     << "' " << context << " but has Token " << curToken;
        if (isprint(curToken))
            llvm:errs() << " '" << (char)curToken << "'";
        llvm::errs() << "\n";
        return nullptr;
    }
};

} // namespace toy end

#endif