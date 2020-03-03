function log_evidence = ldae_is_variants(words, topics, topic_prior, num_samples, variant, variant_iters)
%LDAE_IS_VARIANTS approximate evidence of LDA model, importance sampling with choice of q-distributions
%
% log_evidence = ldae_is_variants(words, topics, topic_prior[, num_samples=1000[, variant=3[, variant_iters=1]]]);
%
% Inputs:
%             words 1xNd
%            topics TxV each row is a distribution over a vocabulary of size V 
%       topic_prior 1xT parameters of Dirichlet from which document topic vector is drawn
%       num_samples 1x1 default 1000
%           variant 1x1 1: prior, 2: q(z) = \prod p(z_n|w_n), 3: q(z) = \prod(z_n|w_n,hacky_pseudo_counts)
%                       Default is 3, as that seems to work best.
%     variant_iters 1x1 If variant needs iterative updates, use this number
%
% Outputs:
%     log_evidence  1x1 

% Iain Murray, January 2009

[T, V] = size(topics);
Nd = length(words);

if ~exist('num_samples', 'var')
    num_samples = 1000;
end
if ~exist('variant', 'var')
    variant = 3;
end
if ~exist('variant_iters', 'var')
    variant_iters = 1;
end

// The dimension of the new matrix is T X 1
// equivalent to alpha * m
topic_prior = topic_prior(:);

topic_alpha = sum(topic_prior); 
%topic_mean = topic_prior / topic_alpha;

if variant == 1
    % Importance sample from prior
    qstar = repmat(topic_prior, 1, Nd); % T x Nd 
    qq = bsxfun(@rdivide, qstar, sum(qstar, 1));  // normalize
else
    //Take w_n into account when picking z_n
    qstar = bsxfun(@times, topic_prior, topics(:, words)); % T x Nd // phi, alpha*m
    qq = bsxfun(@rdivide, qstar, sum(qstar, 1)); // normalize

    if variant == 3
        for i = 1:variant_iters
            % Now create pseudo-counts from qq and recompute qq using them
            pseudo_counts = bsxfun(@minus, topic_prior + sum(qq, 2), qq); // finding the pseudo counts
            qstar = bsxfun(@times, pseudo_counts, topics(:, words)); % T x Nd
            qq = bsxfun(@rdivide, qstar, sum(qstar, 1));
        end
    end
end

% Draw samples from the q-distribution
samples = zeros(Nd, num_samples); // Nd X 1000
for n = 1:Nd // for every word
    samples(n, :) = discreternd(num_samples, qq(:, n)); % Nd x num_samples // sample 1000 times from some discrete distribution for one word // samples the topic for corresponding to a word
end 

// Evaluate P(z, v) at samples and compare to q-distribution
// no of samples belonging to each topic.
Nk = histc(samples, 1:T, 1); // T x num_samples
// log(gamma(T X No_samples + T X 1)) + log(gamma(topic_alpha)) - log(gamma(topic_prior)) - log(gamma(Nd + topic_alpha))
// some normalization constant?
log_pz = sum(gammaln(bsxfun(@plus, Nk, topic_prior)), 1) ...
        + gammaln(topic_alpha) - sum(gammaln(topic_prior)) ...
        - gammaln(Nd + topic_alpha); // 1 x num_samples
log_w_given_z = zeros(1, num_samples); // P(w/z)
for n = 1:Nd // for each word, select all the samples * each word
    // P(w/z) = table lookup
    // product of P(w/z) = sum(log(P(w/z)))
    log_w_given_z = log_w_given_z + log(topics(samples(n,:), words(n)));
end
// log(P(w/z)) + log(P(z))
log_joint = log_pz + log_w_given_z;

%

// 1*num_samples
log_qq = zeros(1, num_samples);
for n = 1:Nd // sample only the topic for the dividor
    log_qq = log_qq + log(qq(samples(n,:), n));
end
log_weights = log_joint - log_qq;
// equivalent to normalizing over all the samples
// log(exp(s1) + exp(s2) + exp(s1000))
log_evidence = logsumexp(log_weights(:)) - log(length(log_weights));