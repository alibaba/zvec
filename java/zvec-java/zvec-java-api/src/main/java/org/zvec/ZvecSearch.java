package org.zvec;

import java.util.Objects;

public final class ZvecSearch {
  private ZvecSearch() {}

  public static Builder vector(String fieldName, float[] queryVector) {
    return new Builder(VectorQuery.of(fieldName, queryVector));
  }

  public static final class Builder {
    private final VectorQuery query;

    private Builder(VectorQuery query) {
      this.query = Objects.requireNonNull(query, "query");
    }

    public Builder topK(int topK) {
      query.topK(topK);
      return this;
    }

    public Builder fast() {
      applyProfile(TuningProfile.FAST);
      return this;
    }

    public Builder balanced() {
      applyProfile(TuningProfile.BALANCED);
      return this;
    }

    public Builder accurate() {
      applyProfile(TuningProfile.ACCURATE);
      return this;
    }

    public Builder project(String... fields) {
      query.outputFields(fields);
      return this;
    }

    public Builder includeVector() {
      query.includeVector(true);
      return this;
    }

    public Builder filter(String filter) {
      query.filter(filter);
      return this;
    }

    public VectorQuery build() {
      return query;
    }

    private void applyProfile(TuningProfile profile) {
      query.withTuningProfile(profile);
    }
  }
}
