// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------
#include "learner.h"
#include "InstanceBasedSupervisedLearner.h"
#include "DecisionTreeLearner.h"
#include "PerceptronLearner.h"
#include "PerceptronNeuronBankFactory.h"
#include "BackPropNeuronBankFactory.h"
#include "baseline.h"
#include "error.h"
#include "rand.h"
#include "filter.h"
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <algorithm>
#include <sstream>
#include <memory>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <cctype>
#ifdef WIN32
# include <windows.h>
# include <float.h>
#else // WIN32
# include <sys/time.h>
# include <fenv.h>
#endif // else WIN32


using namespace std;
using namespace ml;

void ExtractRealValueError(const string & name)
{
    stringstream ss;
    ss << name << " must be a valid floating-point number "
       << "and should be between 0 and 1." << endl
       << "For validation purposes, this number should not be "\
       << "specified in scientific notation, " << endl
       << "But should be of the form /\\+?0?.[0-9]*/ .";
    ThrowError(ss.str().c_str());
}

double ExtractRealValue(const char * nstr, const string & name)
{
    string number_string = nstr;
    bool IsDigit = true;
    string::iterator j = number_string.begin();

    if (*j == '+')
    {
        j++;
    }

    if (j == number_string.end())
    {
        ExtractRealValueError(name);
    }
    else if (*j == '0')
    {
        j++;
    }

    if (j == number_string.end())
    {
        ExtractRealValueError(name);
    }
    else if (*j == '.')
    {
        j++;
    }
    else
    {
    }

    for ( ; j != number_string.end(); j++)
    {
        if (!isdigit(*j))
        {
            IsDigit= false;
            break;
        }
    }

    if (!IsDigit)
    {
        ExtractRealValueError(name);
    }

    stringstream ss(number_string);
    double returned;
    ss >> returned;
    return returned;
}

class ArgParser
{
    string arff;
    string learner;
    string learning_rate_string;
    string evaluation;
    char* evalExtra;
    bool normalize;
    bool nominal_to_cat;
    bool discretize;
    unsigned int seed;
    double learning_rate;
    double momentum_term;
    string momentum_term_string;
    bool is_nominal;
    std::vector<int> levels;
    int k;
    bool inverse_square_weight;
    double reduction;

public:
    //You may need to add more options for specific learning models
    ArgParser ( char* argv[], int argc ) : levels()
    {
        arff = "";
        learner = "";
        evaluation = "";
        seed = (unsigned int)time ( NULL );
        normalize = false;
        nominal_to_cat = false;
        discretize = false;
        learning_rate = 0.1;
        is_nominal = false;
        inverse_square_weight = false;
        reduction = 0.0;
        k = 1;
        for ( int i = 1; i < argc; i++ )
        {
            if ( strcmp ( argv[i], "-A" ) == 0 )
            {
                arff = argv[++i];
            }
            else if ( strcmp ( argv[i], "-L" ) == 0 )
            {
                learner = argv[++i];
                if (learner == "neuralnet")
                {
                    if (strcmp(argv[++i], "--nominal") == 0)
                    {
                        is_nominal = true;
                        i++;
                    }
                    int size = atoi(argv[i]);
                    if (size < 0)
                    {
                        ThrowError("number of neural net levels must not be negative.");
                    }

                    levels.reserve(size);
                    for (int j = 0; j < size; j++)
                    {
                        levels.push_back(atoi(argv[++i]));
                    }
                }
            }
            else if ( strcmp ( argv[i], "-E" ) == 0 )
            {
                evaluation = argv[++i];
                if ( strcmp ( argv[i], "static" ) == 0 )
                    evalExtra = argv[++i]; //expecting a test set name
                else if ( strcmp ( argv[i], "random" ) == 0 )
                    evalExtra = argv[++i]; //expecting a double representing the percentage for testing. Note stratification is NOT done
                else if ( strcmp ( argv[i], "cross" ) == 0 )

                    evalExtra = argv[++i]; //expecting the number of folds
                else if ( strcmp ( argv[i], "training" ) != 0 )
                    ThrowError ( "Invalid Evaluation Method: ", argv[i] );
            }
            else if ( strcmp ( argv[i], "-N" ) == 0 )
                normalize = true;
            else if ( strcmp ( argv[i], "-C" ) == 0 )
                nominal_to_cat = true;
            else if ( strcmp ( argv[i], "-D" ) == 0 )
                discretize = true;
            else if ( strcmp ( argv[i], "-R" ) == 0 )
                seed = atoi ( argv[++i] );
            else if ( strcmp ( argv[i], "-i" ) == 0 )
                inverse_square_weight = true;
            else if ( strcmp ( argv[i], "-k" ) == 0 )
                k = atoi (argv [++i] );
            else if ( strcmp ( argv[i], "-r" ) == 0 )
            {
                reduction = 
                    ExtractRealValue(argv[++i], std::string("Reduction"));

            }
            else if ( strcmp ( argv[i], "-l" ) == 0 )
            {
                learning_rate = 
                    ExtractRealValue(argv[++i], std::string("Learning Rate"));
            }
            else if ( strcmp ( argv[i], "-m" ) == 0 )
            {
                momentum_term = 
                    ExtractRealValue(argv[++i], std::string("Momentum Term"));
            }
            else
                ThrowError ( "Invalid paramater: ", argv[i] );
        }
        if ( arff == "" || learner == "" || evaluation == "" )
        {
            cout << "Missing parameters.  Usage:\n"
                 << "MLSystemManager -L <learningAlgorithm> -A <ARFF_File> " << endl
                 << "    -E <evaluationMethod> [<EvaluationParams>] " << endl
                 << "    [-N] [-C] [-D] [-R <seed>] [-l <learningRate>] [-m <momentumTerm>]" << endl
                 << endl
                 << "Options: " << endl
                 << "-L <learningAlgorithm>    Specify the learning algorithm to use " << endl
                 << "-A <ARFF_FILE>            Specify the ARFF_FILE to use for learning" << endl
                 << "-E <evaluationMethod>     Specify which evaluation method to use" << endl
                 << "-N                        Normalize the data (optional)" << endl
                 << "-C                        \"Nominal to Categorical\"ize the data (optional)" << endl
                 << "-D                        Discretize the data (optional)" << endl
                 << "-R <seed>                 Specifty the random seed to use (optional)" << endl
                 << "-l <learningRate>         For perceptron and neuralnet, Specify the learning rate (defaults to .1)" << endl
                 << "-m <momentumTerm>         For perceptron and neuralnet, Specify the momentum term (defaults to .9)" << endl
                 << "-k <k>                    For KNN, Specify the 'k' value" << endl
                 << "-i                        For KNN, Specify that inverse-square weighting" << endl
                 << "-r <val>                  For KNN, Specify the reduction ratio (defaults to 0 - no reduction)" << endl
                 << endl
                 << "Possible values for <evaluationMethod>:" << endl
                 << "    training" << endl
                 << "    static [<TestARFF_File>]" << endl
                 << "    random <PercentageForTesting>" << endl
                 << "    cross  <numOfFolds>" << endl
                 << endl
                 << "Possible values for <learningAlgorithm>: " << endl
                 << "    baseline" << endl
                 << "    perceptron" << endl
                 << "    neuralnet" << endl
                 << "    decisiontree" << endl
                 << "    naivebayes" << endl
                 << "    knn" << endl
                 << endl
                 << "If 'neuralnet' is used for <learningAlgorithm>, the following" << endl
                 << "  arguments are required: " << endl
                 << "    [--nominal] <num-levels> <level-size> [<level-size> ...]" << endl
                 << endl;
            ThrowError ( "Missing parameters" );
        }
    }

    //The getter methods
    string getARFF()
    {
        return arff;
    }
    string getLearner()
    {
        return learner;
    }
    string getEvaluation()
    {
        return evaluation;
    }
    char* getEvalExtra()
    {
        return evalExtra;
    }
    bool getNormal()
    {
        return normalize;
    }
    
    bool getInverseSquareWeight()
    {
        return inverse_square_weight;
    }

    bool getNominalToCat()
    {
        return nominal_to_cat;
    }
    bool getDiscretize()
    {
        return discretize;
    }
    unsigned int getSeed()
    {
        return seed;
    }
    int getK()
    {
        return k;
    }

    double getReductionRate()
    {
        return reduction;
    }

    double getLearningRate()
    {
        return learning_rate;
    }
    double getMomentumTerm()
    {
        return momentum_term;
    }
    bool getIsNominal()
    {
        return is_nominal;
    }
    const std::vector<int> & getLevels()
    {
        return levels;
    }
};

// Returns the number of seconds since some fixed point in the past, with at least millisecond precision
double getTime()
{
#ifdef WIN32
    time_t t;
    SYSTEMTIME st;
    GetSystemTime(&st);
    return ((double)st.wMilliseconds * 1e-3 + time(&t));
#else
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1e-6);
#endif
}

SupervisedLearner* getLearner(string model, Rand& r, double learn, double momentum, bool is_nominal, const vector<int> & levels, int k, bool InverseSquare,
        double ReductionTerm)
{
    if (model.compare("baseline") == 0)
        return new BaselineLearner(r);
    else if (model.compare("perceptron") == 0)
    {
        return new PerceptronLearner(r,learn,momentum,
            PerceptronNeuronBankFactory::CreateInstance());
    }
    else if (model.compare("neuralnet") == 0)
    {
        return new PerceptronLearner(r,learn,momentum,
            BackPropNeuronBankFactory::CreateInstance(is_nominal, levels));
    }
    else if (model.compare("decisiontree") == 0)
    {
        return new DecisionTreeLearner(r);
    }
    else if (model.compare("naivebayes") == 0)
        ThrowError("Sorry, ", model, " is not yet implemented");
    else if (model.compare("knn") == 0)
    {
        return new InstanceBasedSupervisedLearner(r,k, InverseSquare, ReductionTerm);
    }
    else
        ThrowError("Unrecognized model: ", model);
    return NULL;
}

void doit(ArgParser& parser)
{
    // Load the model
    Rand r ( parser.getSeed() );
    string model = parser.getLearner();
    SupervisedLearner* learner = getLearner(model, r,
        parser.getLearningRate(), parser.getMomentumTerm(),
        parser.getIsNominal(), parser.getLevels(),
        parser.getK(), parser.getInverseSquareWeight(), 
        parser.getReductionRate());

    // Wrap the learner with the specified filters
    if ( parser.getNominalToCat() )
        learner = new NominalToCategorical( learner );
    if ( parser.getNormal() )
        learner = new Normalize( learner );
    if ( parser.getDiscretize() )
        learner = new Discretize( learner );

    // This line says to "delete(learner)" when this object (ap_learner) goes out of scope. This
    // technique is better than just calling "delete(learner)" ourselves at the end
    // of this method because this will clean up memory even if an exception is thrown.
    shared_ptr<SupervisedLearner> ap_learner ( learner );

    // Load the ARFF file
    string fileName = parser.getARFF();
    Matrix dataset;
    dataset.loadARFF ( fileName );
    size_t labelDims = 1;

    // Display some values
    cout << "Dataset name: " << fileName << endl;
    cout << "Number of instances (rows): " << dataset.rows() << endl;
    cout << "Number of attributes (cols): " << dataset.cols() << endl;
    cout << "Learning algorithm: " << model << endl;
    string evaluation = parser.getEvaluation();
    cout << "Evaluation method: " << evaluation << endl;
    if ( evaluation.compare ( "training" ) == 0 )
    {
        // Train
        Matrix trainFeatures, trainLabels;
        trainFeatures.copyPart(dataset, 0, 0, dataset.rows(), dataset.cols() - labelDims);
        trainLabels.copyPart(dataset, 0, dataset.cols() - labelDims, dataset.rows(), labelDims);
        double timeBeforeTraining = getTime();
        learner->train(trainFeatures, trainLabels);
        double timeAfterTraining = getTime();

        // Test on the same dataset
        Matrix stats;
        double timeBeforeTesting = getTime();
        double accuracy = learner->measureAccuracy(trainFeatures, trainLabels, &stats);
        double timeAfterTesting = getTime();

        // Print results
        cout << "\n\nAccuracy on the training set: (does NOT imply the ability to generalize)\n";
        for(size_t i = 0; i < stats.cols(); i++)
            cout << dataset.attrValue(dataset.cols() - 1, i) << ": " << stats[0][i] << "/" << stats[1][i] << "\n";
        cout << "Set accuracy: " << accuracy << "\n";
        cout<< "\nTraining time: " << (timeAfterTraining - timeBeforeTraining) << " seconds\n";
        cout<< "\nTesting time: " << (timeAfterTesting - timeBeforeTesting) << " seconds\n";
        cout.flush();
    }
    else if ( evaluation.compare ( "static" ) == 0 )
    {
        // Train
        Matrix trainFeatures, trainLabels;
        trainFeatures.copyPart(dataset, 0, 0, dataset.rows(), dataset.cols() - labelDims);
        trainLabels.copyPart(dataset, 0, dataset.cols() - labelDims, dataset.rows(), labelDims);
        double timeBeforeTraining = getTime();
        learner->train(trainFeatures, trainLabels);
        double timeAfterTraining = getTime();

        // Test on the same dataset
        Matrix stats;
        double accuracy = learner->measureAccuracy(trainFeatures, trainLabels, &stats);

        // Print results
        cout << "\n\nAccuracy on the training set: (does NOT imply the ability to generalize)\n";
        for(size_t i = 0; i < stats.cols(); i++)
            cout << dataset.attrValue(dataset.cols() - 1, i) << ": " << stats[0][i] << "/" << stats[1][i] << "\n";
        cout << "Set accuracy: " << accuracy << "\n";
        cout.flush();

        // Test on the test set
        if(!parser.getEvalExtra())
            ThrowError("Expected a test dataset to be specified");
        string testSetFilename = parser.getEvalExtra();
        Matrix testSet;
        testSet.loadARFF(testSetFilename);
        dataset.checkCompatibility(testSet);
        Matrix testFeatures, testLabels;
        testFeatures.copyPart(testSet, 0, 0, testSet.rows(), testSet.cols() - labelDims);
        testLabels.copyPart(testSet, 0, testSet.cols() - labelDims, testSet.rows(), labelDims);
        double timeBeforeTesting = getTime();
        accuracy = learner->measureAccuracy(testFeatures, testLabels, &stats);
        double timeAfterTesting = getTime();

        // Print results
        cout << "\n\nAccuracy on the test set:\n";
        for(size_t i = 0; i < stats.cols(); i++)
            cout << dataset.attrValue(dataset.cols() - 1, i) << ": " << stats[0][i] << "/" << stats[1][i] << "\n";
        cout << "Set accuracy: " << accuracy << "\n";
        cout << "\nTraining time: " << (timeAfterTraining - timeBeforeTraining) << " seconds\n";
        cout << "\nTesting time: " << (timeAfterTesting - timeBeforeTesting) << " seconds\n";
        cout.flush();
    }
    else if ( evaluation.compare ( "random" ) == 0 )
    {
        // Split the data
        if(!parser.getEvalExtra())
            ThrowError("Expected the training percentage to be specified");
        double trainPercent = atof ( parser.getEvalExtra() );
        if ( trainPercent < 0.0 || trainPercent > 1.0 )
            ThrowError("Expected the percentage to be between 0 and 1\n");
        size_t trainRows = (size_t)floor(dataset.rows() * trainPercent + 0.5);
        dataset.shuffleRows(r);
        Matrix trainSet, testSet;
        trainSet.copyPart(dataset, 0, 0, trainRows, dataset.cols());
        testSet.copyPart(dataset, trainRows, 0, dataset.rows() - trainRows, dataset.cols());

        // Train
        Matrix trainFeatures, trainLabels;
        trainFeatures.copyPart(trainSet, 0, 0, trainSet.rows(), trainSet.cols() - labelDims);
        trainLabels.copyPart(trainSet, 0, trainSet.cols() - labelDims, trainSet.rows(), labelDims);
        Matrix testFeatures, testLabels;
        testFeatures.copyPart(testSet, 0, 0, testSet.rows(), testSet.cols() - labelDims);
        testLabels.copyPart(testSet, 0, testSet.cols() - labelDims, testSet.rows(), labelDims);
        double timeBeforeTraining = getTime();
        learner->train(trainFeatures, trainLabels, &testFeatures,
                &testLabels);
        double timeAfterTraining = getTime();

        // Test on the same dataset
        Matrix stats;
        double accuracy = learner->measureAccuracy(trainFeatures, trainLabels, &stats);

        // Print results
        cout << "\n\nAccuracy on the training set: (does NOT imply the ability to generalize)\n";
        for(size_t i = 0; i < stats.cols(); i++)
            cout << dataset.attrValue(dataset.cols() - 1, i) << ": " << stats[0][i] << "/" << stats[1][i] << "\n";
        cout << "Set accuracy: " << accuracy << "\n";
        cout.flush();

        // Test on the test set
        double timeBeforeTesting = getTime();
        accuracy = learner->measureAccuracy(testFeatures, testLabels, &stats);
        double timeAfterTesting = getTime();

        // Print results
        cout << "\n\nAccuracy on the test set:\n";
        for(size_t i = 0; i < stats.cols(); i++)
            cout << dataset.attrValue(dataset.cols() - 1, i) << ": " << stats[0][i] << "/" << stats[1][i] << "\n";
        cout << "Set accuracy: " << accuracy << "\n";
        cout<< "\nTraining time: " << (timeAfterTraining - timeBeforeTraining) << " seconds\n";
        cout<< "\nTesting time: " << (timeAfterTesting - timeBeforeTesting) << " seconds\n";
        cout.flush();
    }
    else if ( evaluation.compare ( "cross" ) == 0 )
    {
        if(!parser.getEvalExtra())
            ThrowError("Expected the number of folds to be specified");
        size_t folds = atoi ( parser.getEvalExtra() );
        Matrix features, labels;
        features.copyPart(dataset, 0, 0, dataset.rows(), dataset.cols() - labelDims);
        labels.copyPart(dataset, 0, dataset.cols() - labelDims, dataset.rows(), labelDims);
        double accuracy = learner->crossValidate(1, folds, features, labels, r, true);
        if(labels.valueCount(0) == 0)
            cout << "Root Mean Squared Error: " << accuracy << "\n";
        else
            cout << "Mean predictive accuracy: " << accuracy << "\n";
        cout.flush();
    }
}

int main(int argc, char *argv[])
{
    srand(time(NULL));
    // Swallowing errors leads to painful debugging experiences--never swallow errors.
#ifdef WIN32
    // Don't silently swallow floating point errors on Windows.
    unsigned int cw = _control87(0, 0) & MCW_EM;
    cw &= ~(_EM_INVALID | _EM_ZERODIVIDE | _EM_OVERFLOW);
    _control87(cw, MCW_EM);
#else
#   ifdef DARWIN
    // Anyone know how to tell Darwin to break if there is a floating point error?
#   else
    // Don't silently swallow floating point errors on Linux.
    feenableexcept(FE_INVALID | FE_DIVBYZERO | FE_OVERFLOW);
#   endif
#endif

    int ret = 0;
    try
    {
        // parse the args
        ArgParser parser(argv, argc);

        // do what they say to do
        doit(parser);
    }
    catch(const std::exception& e)
    {
        cerr << "Error: " << e.what() << "\n";
        ret = 1;
    }

    return ret;
}
