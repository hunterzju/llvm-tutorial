#include "./include/KaleidoscopeJIT.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/InstCombine/InstCombine.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace llvm;
using namespace llvm::orc;

// Lexer
//
enum Token {
	tok_eof = -1,
	// commands
	tok_def = -2,
	tok_extern = -3,
	// primary
	tok_identifier = -4,
	tok_number = -5,

	// control 
	tok_if = -6,
	tok_then = -7,
	tok_else = -8,
	tok_for = -9,
	tok_in = -10,

	// operators
	tok_binary = -11,
	tok_unary = -12
};

static std::string IdentifierStr;
static double NumVal;

static int gettok() {
	static int LastChar = ' ';

	while (isspace(LastChar))
		LastChar = getchar();

	if (isalpha(LastChar)) {
		IdentifierStr = LastChar;
		while(isalnum(LastChar = getchar()))
			IdentifierStr += LastChar;

		if (IdentifierStr == "def")
			return tok_def;
		if (IdentifierStr == "extern")
			return tok_extern;
		if (IdentifierStr == "if")
			return tok_if;
		if (IdentifierStr == "then")
			return tok_then;
		if (IdentifierStr == "else")
			return tok_else;
		if (IdentifierStr == "for")
			return tok_for;
		if (IdentifierStr == "in")
			return tok_in;
		if (IdentifierStr == "binary")
			return tok_binary;
		if (IdentifierStr == "unary")
			return tok_unary;
		return tok_identifier;
	}

	if (isdigit(LastChar) || LastChar == '.') {
		std::string NumStr;
		do {
			NumStr += LastChar;
			LastChar = getchar();
		} while (isdigit(LastChar) || LastChar == '.');

		NumVal = strtod(NumStr.c_str(), nullptr);
		return  tok_number;
	}

	if (LastChar == '#') {
		do
			LastChar = getchar();
		while (LastChar != EOF && LastChar != '\n' && LastChar != '\r');

		if (LastChar != EOF)
			return gettok();
	}

	if (LastChar == EOF)
		return tok_eof;

	int ThisChar = LastChar;
	LastChar = getchar();
	return ThisChar;
}

// Abstract Syntax Tree
namespace {

// ExprAst
class ExprAST {
public:
	virtual ~ExprAST() = default;
	virtual Value *codegen() = 0;
};

class NumberExprAST : public ExprAST {
	double Val;

public:
	NumberExprAST(double Val) : Val(Val) {}
	Value *codegen() override;
};

class VariableExprAST : public ExprAST {
	std::string Name;

public:
	VariableExprAST(const std::string &Name) : Name(Name) {}
	Value *codegen() override;
};

class UnaryExprAST : public ExprAST {
	char Opcode;
	std::unique_ptr<ExprAST> Operand;

public:
	UnaryExprAST(char Opcode, std::unique_ptr<ExprAST> Operand)
		: Opcode(Opcode), Operand(std::move(Operand)) {}
	
	Value *codegen() override;
};

class BinaryExprAST : public ExprAST {
	char Op;
	std::unique_ptr<ExprAST> LHS, RHS;

public:
	BinaryExprAST(char Op, std::unique_ptr<ExprAST> LHS,
				  std::unique_ptr<ExprAST> RHS)
		: Op(Op), LHS(std::move(LHS)), RHS(std::move(RHS)) {}
	Value *codegen() override;
};

class CallExprAST : public ExprAST {
	std::string Callee;
	std::vector<std::unique_ptr<ExprAST>> Args;

public:
	CallExprAST(const std::string &Callee,
				std::vector<std::unique_ptr<ExprAST>> Args)
		: Callee(Callee), Args(std::move(Args)) {}
	Value *codegen() override;
};

// IfExprAST
class IfExprAST : public ExprAST {
	std::unique_ptr<ExprAST> Cond, Then, Else;

public:
	IfExprAST(std::unique_ptr<ExprAST> Cond, std::unique_ptr<ExprAST> Then,
			  std::unique_ptr<ExprAST> Else)
		: Cond(std::move(Cond)), Then(std::move(Then)), Else(std::move(Else))
		{}
	
	Value *codegen() override;
};

// ForExprAST
class ForExprAST : public ExprAST {
	std::string VarName;
	std::unique_ptr<ExprAST> Start, End, Step, Body;

public:
	ForExprAST(const std::string &VarName, std::unique_ptr<ExprAST> Start,
			   std::unique_ptr<ExprAST> End, std::unique_ptr<ExprAST> Step,
			   std::unique_ptr<ExprAST> Body)
		: VarName(VarName), Start(std::move(Start)), End(std::move(End)),
		  Step(std::move(Step)), Body(std::move(Body))
		  {}
	
	Value *codegen() override;
};

class PrototypeAST {
	std::string Name;
	std::vector<std::string> Args;
	bool IsOperator;
	unsigned Precedence;

public:
	PrototypeAST(const std::string &Name, std::vector<std::string> Args,
				 bool IsOperator = false, unsigned Prec = 0)
		: Name(Name), Args(std::move(Args)), IsOperator(IsOperator),
		  Precedence(Prec) {}

	const std::string &getName() const { return Name;  }
	Function *codegen();

	bool isUnaryOp() const { return IsOperator && Args.size() == 1; }
	bool isBinaryOp() const { return IsOperator && Args.size() == 2; }

	char getOperatorName() const {
		assert(isUnaryOp() || isBinaryOp());
		return Name[Name.size() - 1];
	}

	unsigned getBinaryPrecedence() const { return Precedence; }
};

class FunctionAST {
	std::unique_ptr<PrototypeAST> Proto;
	std::unique_ptr<ExprAST> Body;

public:
	FunctionAST(std::unique_ptr<PrototypeAST> Proto,
				std::unique_ptr<ExprAST> Body)
		: Proto(std::move(Proto)), Body(std::move(Body)) {}
	Function *codegen();
};

} // end anonymous namespace

// Parser
static int CurTok;
static int getNextToken() { return CurTok = gettok(); }

// binary op precedence
static std::map<char, int> BinopPrecedence;

static int GetTokPrecedence() {
	if (!isascii(CurTok))
		return -1;
	
	int TokPrec = BinopPrecedence[CurTok];
	if (TokPrec <= 0)
		return -1;
	return TokPrec;
}

std::unique_ptr<ExprAST> LogError(const char *Str) {
	fprintf(stderr, "Error: %s\n", Str);
	return nullptr;
}

std::unique_ptr<PrototypeAST> LogErrorP(const char *Str) {
	LogError(Str);
	return nullptr;
}

static std::unique_ptr<ExprAST> ParseExpression();

static std::unique_ptr<ExprAST> ParseNumberExpr() {
	auto Result = std::make_unique<NumberExprAST>(NumVal);
	getNextToken();
	return std::move(Result);
}

static std::unique_ptr<ExprAST> ParseParenExpr() {
	getNextToken(); //eat (
	auto V = ParseExpression();
	if (!V)
		return nullptr;

	if (CurTok != ')')
		return LogError("expected ')'");
	getNextToken();		// eat ')'
	return V;
}

static std::unique_ptr<ExprAST> ParseIdentifierExpr() {
	std::string IdName = IdentifierStr;

	getNextToken();

	if (CurTok != '(')
		return std::make_unique<VariableExprAST>(IdName);

	getNextToken();
	std::vector<std::unique_ptr<ExprAST>> Args;
	if (CurTok != ')') {
		while (true) {
			if (auto Arg = ParseExpression())
				Args.push_back(std::move(Arg));
			else
				return nullptr;
			
			if (CurTok == ')')
				break;
			
			if (CurTok != ',')
				return LogError("Expected ')' or ',' in argument list");
			getNextToken();
		}
	}

	getNextToken(); 	// eat ')'

	return std::make_unique<CallExprAST>(IdName, std::move(Args));
}

static std::unique_ptr<ExprAST> ParseIfExpr() {
	getNextToken();

	auto Cond = ParseExpression();
	if (!Cond)
		return nullptr;
	
	if (CurTok != tok_then)
		return LogError("expected then");
	getNextToken();

	auto Then = ParseExpression();
	if (!Then)
		return nullptr;
	
	if (CurTok != tok_else)
		return LogError("expected else");
	getNextToken();

	auto Else = ParseExpression();
	if (!Else)
		return nullptr;

	return std::make_unique<IfExprAST>(std::move(Cond), std::move(Then),
										std::move(Else));
}

static std::unique_ptr<ExprAST> ParseForExpr() {
	getNextToken();

	if (CurTok != tok_identifier)
		return LogError("expected identifier after for");
	
	std::string IdName = IdentifierStr;
	getNextToken();

	if (CurTok != '=')
		return LogError("expected '=' after for");
	getNextToken();

	auto Start = ParseExpression();
	if (!Start)
		return nullptr;
	if (CurTok != ',')
		return LogError("expected ',' after for start value");
	getNextToken();

	auto End = ParseExpression();
	if (!End)
		return nullptr;
	
	std::unique_ptr<ExprAST> Step;
	if (CurTok == ',') {
		getNextToken();
		Step = ParseExpression();
		if (!Step)
			return nullptr;
	}

	if (CurTok != tok_in)
		return LogError("expected 'in' after for");
	getNextToken();

	auto Body = ParseExpression();
	if (!Body)
		return nullptr;
	
	return std::make_unique<ForExprAST>(IdName, std::move(Start), std::move(End),
										std::move(Step), std::move(Body));
}

static std::unique_ptr<ExprAST> ParsePrimary() {
	switch (CurTok) {
		default:
			return LogError("unknown token when expecting an expression");
		case tok_identifier:
			return ParseIdentifierExpr();
		case tok_number:
			return ParseNumberExpr();
		case '(':
			return ParseParenExpr();
		case tok_if:
			return ParseIfExpr();
		case tok_for:
			return ParseForExpr();
	}
}

// unary
static std::unique_ptr<ExprAST> ParseUnary() {
	if (!isascii(CurTok) || CurTok == '(' || CurTok == ',')
		return ParsePrimary();

	int Opc = CurTok;
	getNextToken();
	if (auto Operand = ParseUnary())
		return std::make_unique<UnaryExprAST>(Opc, std::move(Operand));
	return nullptr;
}

// example: 2 + ((3 + 4) * 5 + 6) + 7
static std::unique_ptr<ExprAST> ParseBinOpRHS(int ExprPrec, std::unique_ptr<ExprAST> LHS) {
	while(true) {
		int TokPrec = GetTokPrecedence();

		if (TokPrec < ExprPrec)
			return LHS;
		
		int BinOp = CurTok;
		getNextToken();

		auto RHS = ParseUnary();
		if(!RHS)
			return nullptr;

		int NextPrec = GetTokPrecedence();
		if(TokPrec < NextPrec) {
			RHS = ParseBinOpRHS(TokPrec + 1, std::move(RHS));
			if(!RHS)
				return nullptr;
		}

		LHS = std::make_unique<BinaryExprAST>(BinOp, std::move(LHS), std::move(RHS));
	}
}

static std::unique_ptr<ExprAST> ParseExpression() {
	auto LHS = ParseUnary();
	if (!LHS)
		return nullptr;
	
	return ParseBinOpRHS(0, std::move(LHS));
}

static std::unique_ptr<PrototypeAST> ParsePrototype() {
	std::string FnName;

	unsigned Kind = 0;
	unsigned BinaryPrecedence = 30;

	switch (CurTok)
	{
	default:
		return LogErrorP("Expected function name in prototype");
	case tok_identifier:
		FnName = IdentifierStr;
		Kind = 0;
		getNextToken();
		break;
	case tok_unary:
		getNextToken();
		if (!isascii(CurTok))
			return LogErrorP("Expected unary operator");
		FnName = "unary";
		FnName += (char)CurTok;
		Kind = 1;
		getNextToken();
		break;
	case tok_binary:
		getNextToken();
		if (!isascii(CurTok))
			return LogErrorP("Expected binary operator");
		FnName = "binary";
		FnName += (char)CurTok;
		Kind = 2;
		getNextToken();

		if (CurTok == tok_number) {
			if (NumVal < 1 || NumVal > 100)
				return LogErrorP("Invalid precedence: must be 1..100");
			BinaryPrecedence = (unsigned)NumVal;
			getNextToken();
		}
		break;
	}

	if (CurTok != '(')
		return LogErrorP("Expected '(' in prototype");

	std::vector<std::string> ArgNames;
	while (getNextToken() == tok_identifier)
		ArgNames.push_back(IdentifierStr);
	if (CurTok != ')')
		return LogErrorP("Expected ')' in prototype");

	getNextToken();

	if (Kind && ArgNames.size() != Kind)
		return LogErrorP("Invalid number of operands for operator");

	return std::make_unique<PrototypeAST>(FnName, ArgNames, Kind != 0, BinaryPrecedence);
}

static std::unique_ptr<FunctionAST> ParseDefinition() {
	getNextToken();
	auto Proto = ParsePrototype();
	if (!Proto)
		return nullptr;
	
	if (auto E = ParseExpression())
		return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
	return nullptr;
}

static std::unique_ptr<FunctionAST> ParseTopLevelExpr() {
	if (auto E = ParseExpression()) {
		auto Proto = std::make_unique<PrototypeAST>("__anon_expr",
													std::vector<std::string>());
		return std::make_unique<FunctionAST>(std::move(Proto), std::move(E));
	}
	return nullptr;
}

static std::unique_ptr<PrototypeAST> ParseExtern() {
	getNextToken();
	return ParsePrototype();
}

// Code Generation
//===----------------------------------------------------------------------===//

static std::unique_ptr<LLVMContext> TheContext;
static std::unique_ptr<Module> TheModule;
static std::unique_ptr<IRBuilder<>> Builder;
static std::map<std::string, Value *> NamedValues;
// For OPT and JIT
static std::unique_ptr<legacy::FunctionPassManager> TheFPM;
static std::unique_ptr<KaleidoscopeJIT> TheJIT;
static std::map<std::string, std::unique_ptr<PrototypeAST>> FunctionProtos;
static ExitOnError ExitOnErr;

Value *LogErrorV(const char *Str) {
  LogError(Str);
  return nullptr;
}

Function *getFunction(std::string Name) {
	if (auto *F = TheModule->getFunction(Name))
		return F;
	
	auto FI = FunctionProtos.find(Name);
	if (FI != FunctionProtos.end())
		return FI->second->codegen();

	return nullptr;
}

Value *NumberExprAST::codegen() {
  return ConstantFP::get(*TheContext, APFloat(Val));
}

Value *VariableExprAST::codegen() {
  // Look this variable up in the function.
  Value *V = NamedValues[Name];
  if (!V)
    return LogErrorV("Unknown variable name");
  return V;
}

Value *UnaryExprAST::codegen() {
	Value *OperandV = Operand->codegen();
	if (!OperandV)
		return nullptr;
	
	Function *F = getFunction(std::string("unary") + Opcode);
	if (!F)
		return LogErrorV("Unknown unary operator");
	
	return Builder->CreateCall(F, OperandV, "unop");
}

Value *BinaryExprAST::codegen() {
  Value *L = LHS->codegen();
  Value *R = RHS->codegen();
  if (!L || !R)
    return nullptr;

  switch (Op) {
  case '+':
    return Builder->CreateFAdd(L, R, "addtmp");
  case '-':
    return Builder->CreateFSub(L, R, "subtmp");
  case '*':
    return Builder->CreateFMul(L, R, "multmp");
  case '<':
    L = Builder->CreateFCmpULT(L, R, "cmptmp");
    // Convert bool 0/1 to double 0.0 or 1.0
    return Builder->CreateUIToFP(L, Type::getDoubleTy(*TheContext), "booltmp");
  default:
    break;
  }

  Function *F = getFunction(std::string("binary") + Op);
  assert(F && "binary operator not found!");

  Value *Ops[] = {L, R};
  return Builder->CreateCall(F, Ops, "binop");
}

Value *CallExprAST::codegen() {
  // Look up the name in the global module table.
  Function *CalleeF = TheModule->getFunction(Callee);
  if (!CalleeF)
    return LogErrorV("Unknown function referenced");

  // If argument mismatch error.
  if (CalleeF->arg_size() != Args.size())
    return LogErrorV("Incorrect # arguments passed");

  std::vector<Value *> ArgsV;
  for (unsigned i = 0, e = Args.size(); i != e; ++i) {
    ArgsV.push_back(Args[i]->codegen());
    if (!ArgsV.back())
      return nullptr;
  }

  return Builder->CreateCall(CalleeF, ArgsV, "calltmp");
}

Value *IfExprAST::codegen() {
	Value *CondV = Cond->codegen();
	if (!CondV)
		return nullptr;
	
	CondV = Builder->CreateFCmpONE(
		CondV, ConstantFP::get(*TheContext, APFloat(0.0)), "ifcond"
	);

	Function *TheFunction = Builder->GetInsertBlock()->getParent();

	BasicBlock *ThenBB = BasicBlock::Create(*TheContext, "then", TheFunction);
	BasicBlock *ElseBB = BasicBlock::Create(*TheContext, "else");
	BasicBlock *MergeBB = BasicBlock::Create(*TheContext, "ifcont");

	Builder->CreateCondBr(CondV, ThenBB, ElseBB);

	Builder->SetInsertPoint(ThenBB);
	
	Value *ThenV = Then->codegen();
	if (!ThenV)
		return nullptr;

	Builder->CreateBr(MergeBB);
	ThenBB = Builder->GetInsertBlock();

	TheFunction->getBasicBlockList().push_back(ElseBB);
	Builder->SetInsertPoint(ElseBB);

	Value *ElseV = Else->codegen();
	if (!ElseV)
		return nullptr;

	Builder->CreateBr(MergeBB);
	ElseBB = Builder->GetInsertBlock();

	TheFunction->getBasicBlockList().push_back(MergeBB);
	Builder->SetInsertPoint(MergeBB);
	PHINode *PN = Builder->CreatePHI(Type::getDoubleTy(*TheContext), 2, "iftmp");

	PN->addIncoming(ThenV, ThenBB);
	PN->addIncoming(ElseV, ElseBB);
	return PN;
}

Value *ForExprAST::codegen() {
	Value *StartVal = Start->codegen();
	if (!StartVal)
		return nullptr;
	
	Function *TheFunction = Builder->GetInsertBlock()->getParent();
	BasicBlock *PreheaderBB = Builder->GetInsertBlock();
	BasicBlock *LoopBB = BasicBlock::Create(*TheContext, "loop", TheFunction);

	Builder->CreateBr(LoopBB);
	Builder->SetInsertPoint(LoopBB);

	PHINode *Variable = Builder->CreatePHI(Type::getDoubleTy(*TheContext), 2, VarName);
	Variable->addIncoming(StartVal, PreheaderBB);

	Value *OldVal = NamedValues[VarName];
	NamedValues[VarName] = Variable;

	if (!Body->codegen())
		return nullptr;
	
	Value *StepVal = nullptr;
	if (Step) {
		StepVal = Step->codegen();
		if (!StepVal)
			return nullptr;
	}
	else {
		StepVal = ConstantFP::get(*TheContext, APFloat(1.0));
	}

	Value *NextVar = Builder->CreateFAdd(Variable, StepVal, "nextvar");

	Value *EndCond = End->codegen();
	if (!EndCond)
		return nullptr;

	EndCond = Builder->CreateFCmpONE(
		EndCond, ConstantFP::get(*TheContext, APFloat(1.0)), "loopcond"
	);

	BasicBlock *LoopEndBB = Builder->GetInsertBlock();
	BasicBlock *AfterBB = BasicBlock::Create(*TheContext, "afterloop", TheFunction);

	Builder->CreateCondBr(EndCond, LoopBB, AfterBB);
	Builder->SetInsertPoint(AfterBB);

	Variable->addIncoming(NextVar, LoopEndBB);

	if(OldVal)
		NamedValues[VarName] = OldVal;
	else
		NamedValues.erase(VarName);

	return Constant::getNullValue(Type::getDoubleTy(*TheContext));
}

Function *PrototypeAST::codegen() {
  // Make the function type:  double(double,double) etc.
  std::vector<Type *> Doubles(Args.size(), Type::getDoubleTy(*TheContext));
  FunctionType *FT =
      FunctionType::get(Type::getDoubleTy(*TheContext), Doubles, false);

  Function *F =
      Function::Create(FT, Function::ExternalLinkage, Name, TheModule.get());

  // Set names for all arguments.
  unsigned Idx = 0;
  for (auto &Arg : F->args())
    Arg.setName(Args[Idx++]);

  return F;
}

Function *FunctionAST::codegen() {
  // First, check for an existing function from a previous 'extern' declaration.
  auto &P = *Proto;
  FunctionProtos[Proto->getName()] = std::move(Proto);
  Function *TheFunction = getFunction(P.getName());

  if (!TheFunction)
    return nullptr;

  // Create a new basic block to start insertion into.
  BasicBlock *BB = BasicBlock::Create(*TheContext, "entry", TheFunction);
  Builder->SetInsertPoint(BB);

  // Record the function arguments in the NamedValues map.
  NamedValues.clear();
  for (auto &Arg : TheFunction->args())
    NamedValues[std::string(Arg.getName())] = &Arg;

  if (Value *RetVal = Body->codegen()) {
    // Finish off the function.
    Builder->CreateRet(RetVal);

    // Validate the generated code, checking for consistency.
    verifyFunction(*TheFunction);

	TheFPM->run(*TheFunction);

    return TheFunction;
  }

  // Error reading body, remove function.
  TheFunction->eraseFromParent();

  if (P.isBinaryOp())
	BinopPrecedence.erase(P.getOperatorName());
  return nullptr;
}


// Top-level parsing and JIT Driver

static void InitializeModuleAndPassManager() {
	TheContext = std::make_unique<LLVMContext>();
	TheModule = std::make_unique<Module>("my cool jit", *TheContext);
	TheModule->setDataLayout(TheJIT->getDataLayout());

	Builder = std::make_unique<IRBuilder<>>(*TheContext);

	TheFPM = std::make_unique<legacy::FunctionPassManager>(TheModule.get());

	TheFPM->add(createInstructionCombiningPass());
	TheFPM->add(createReassociatePass());
	TheFPM->add(createGVNPass());
	TheFPM->add(createCFGSimplificationPass());

	TheFPM->doInitialization();
}

static void HandleDefinition() {
	if (auto FnAST = ParseDefinition()) {
		if (auto *FnIR = FnAST->codegen()) {
			fprintf(stderr, "Read function definition:");
			FnIR->print(errs());
			fprintf(stderr, "\n");
			ExitOnErr(TheJIT->addModule(
				ThreadSafeModule(std::move(TheModule), std::move(TheContext))
			));
			InitializeModuleAndPassManager();
		}
	}
	else {
		getNextToken();
	}
}

static void HandleExtern() {
	if (auto ProtoAST = ParseExtern()) {
		if (auto *FnIR = ProtoAST->codegen()) {
			fprintf(stderr, "Read extern: ");
			FnIR->print(errs());
			fprintf(stderr, "\n");
			FunctionProtos[ProtoAST->getName()] = std::move(ProtoAST);
		}	
	}
	else {
		getNextToken();
	}
}

static void HanldeTopLevelExpression() {
	if (auto FnAST = ParseTopLevelExpr()) {
		if (FnAST->codegen()) {
			auto RT = TheJIT->getMainJITDylib().createResourceTracker();
			
			auto TSM = ThreadSafeModule(std::move(TheModule), std::move(TheContext));
			ExitOnErr(TheJIT->addModule(std::move(TSM), RT));
			InitializeModuleAndPassManager();

			auto ExprSymbol = ExitOnErr(TheJIT->lookup("__anon_expr"));

			double (*FP)() = (double (*)())(intptr_t)ExprSymbol.getAddress();
			fprintf(stderr, "Evaluated to %f\n", FP());
			
			ExitOnErr(RT->remove());
		}
	}
	else {
		getNextToken();
	}
}

static void MainLoop() {
	while (true) {
		fprintf(stderr, "ready> ");
		switch (CurTok)
		{
		case tok_eof:
			return;
		case ';':
			getNextToken();
			break;
		case tok_def:
			HandleDefinition();
			break;
		case tok_extern:
			HandleExtern();
			break;
		default:
			HanldeTopLevelExpression();
			break;
		}
	}
}

// "Library" functions can be "extern'd" from user code.
#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" DLLEXPORT double putchard(double X) {
	fputc((char)X, stderr);
	return 0;
}

extern "C" DLLEXPORT double printd(double X) {
	fprintf(stderr, "%f\n", X);
	return 0;
}


int main() {
	InitializeNativeTarget();
	InitializeNativeTargetAsmPrinter();
	InitializeNativeTargetAsmParser();

	BinopPrecedence['<'] = 10;
	BinopPrecedence['+'] = 20;
	BinopPrecedence['-'] = 20;
	BinopPrecedence['*'] = 40;

	fprintf(stderr, "ready> ");
	getNextToken();

	TheJIT = ExitOnErr(KaleidoscopeJIT::Create());

	InitializeModuleAndPassManager();

	MainLoop();

	// TheModule->print(errs(), nullptr);

	return 0;
}
